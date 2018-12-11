from __future__ import division
import logging
import os
import random
import time
import shutil

import torch
import torchtext
from torch import optim

from collections import defaultdict

import machine
from machine.evaluator import Evaluator
from machine.loss import NLLLoss
from machine.metrics import WordAccuracy
from machine.optim import Optimizer
from machine.util.checkpoint import Checkpoint
from machine.util.callbacks import CallbackContainer, Logger, ModelCheckpoint
from machine.util.log import Log

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SupervisedTrainer(object):
    """ The SupervisedTrainer class helps in setting up a training framework in a
    supervised setting.

    Args:
        expt_dir (optional, str): experiment Directory to store details of the experiment,
            by default it makes a folder in the current directory to store the details (default: `experiment`).
        loss (list, optional): list of machine.loss.Loss objects for training (default: [machine.loss.NLLLoss])
        metrics (list, optional): list of machine.metric.metric objects to be computed during evaluation
        batch_size (int, optional): batch size for experiment, (default: 64)
        checkpoint_every (int, optional): number of epochs to checkpoint after, (default: 100)
        print_every (int, optional): number of iterations to print after, (default: 100)
    """

    def __init__(self, expt_dir='experiment', loss=[NLLLoss()],
                 loss_weights=None, metrics=[], batch_size=64,
                 eval_batch_size=128, random_seed=None,
                 checkpoint_every=100, print_every=100):
        self._trainer = "Simple Trainer"

        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)

        self.loss = loss
        self.metrics = metrics
        self.loss_weights = loss_weights or len(loss)*[1.]
        self.evaluator = Evaluator(
            loss=self.loss, metrics=self.metrics, batch_size=eval_batch_size)
        self.optimizer = None
        self.checkpoint_every = checkpoint_every
        self.print_every = print_every

        if not os.path.isabs(expt_dir):
            expt_dir = os.path.join(os.getcwd(), expt_dir)
        self.expt_dir = expt_dir
        if not os.path.exists(self.expt_dir):
            os.makedirs(self.expt_dir)
        self.batch_size = batch_size

        self.logger = logging.getLogger(__name__)

    def _train_batch(self, input_variable, input_lengths, target_variable, teacher_forcing_ratio):
        loss = self.loss

        # Forward propagation
        decoder_outputs, decoder_hidden, other = self.model(
            input_variable, input_lengths, target_variable, teacher_forcing_ratio=teacher_forcing_ratio)

        losses = self.evaluator.compute_batch_loss(
            decoder_outputs, decoder_hidden, other, target_variable)

        # Backward propagation
        for i, loss in enumerate(losses, 0):
            loss.scale_loss(self.loss_weights[i])
            loss.backward(retain_graph=True)
        self.optimizer.step()
        self.model.zero_grad()

        return losses

    def _train_epoches(self, data, n_epochs,
                       start_epoch, start_step,
                       callbacks,
                       dev_data=None, monitor_data=[],
                       teacher_forcing_ratio=0):

        # TODO: move this out of the train_epoches function
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=False, sort_within_batch=True,
            sort_key=lambda x: len(x.src),
            device=device, repeat=False)

        steps_per_epoch = len(batch_iterator)
        total_steps = steps_per_epoch * n_epochs

        # give start information to callbacks
        callbacks.set_info(start_step, start_epoch,
                           steps_per_epoch,
                           total_steps)

        # set data as attribute to trainer
        self.data = data
        self.val_data = dev_data or data
        self.monitor_data = monitor_data

        ########################################

        # Call all callbacks
        # set the best losses based on starting accuracies
        self.losses, self.metrics = self.evaluator.evaluate(self.model,
                                                            self.val_data,
                                                            self.get_batch_data)

        callbacks.on_train_begin()
        # TODO check if checkpoint is saved correctly in Checkpoint callback

        # TODO this should also be callback
        logs = Log()

        for epoch in range(start_epoch, n_epochs + 1):

            # call all callbacks,
            # TODO describe which are the default ones
            callbacks.on_epoch_begin(epoch)

            ##########################################
            # TODO this does not seem needed, remove this bit
            batch_generator = batch_iterator.__iter__()

            # consuming seen batches from previous training
            for _ in range((epoch - 1) * steps_per_epoch, start_step):
                next(batch_generator)
            #
            ##########################################

            self.model.train()

            for batch in batch_generator:

                callbacks.on_batch_begin(batch)

                input_variables, input_lengths, target_variables = self.get_batch_data(
                    batch)

                # TODO this should be moved to a "train" callback, this callback
                # should interact with both trainer.model and trainer.val_losses
                # etc so that its outcomes are also accessible for the logger
                self.losses = self._train_batch(input_variables,
                                                input_lengths.tolist(),
                                                target_variables,
                                                teacher_forcing_ratio)

                #########################################
                # TODO this is unrevised, should be checked

                callbacks.on_batch_end(batch)

            #
            ############################################

            callbacks.on_epoch_end(epoch)

        callbacks.on_train_end()

        return logs

    def train(self, model, data, num_epochs=5,
              resume=False, dev_data=None,
              monitor_data={}, optimizer=None,
              teacher_forcing_ratio=0,
              custom_callbacks=[],
              learning_rate=0.001, checkpoint_path=None, top_k=5):
        """ Run training for a given model.

        Args:
            model (machine.models): model to run training on, if `resume=True`, it would be
               overwritten by the model loaded from the latest checkpoint.
            data (machine.dataset.dataset.Dataset): dataset object to train on
            num_epochs (int, optional): number of epochs to run (default 5)
            resume(bool, optional): resume training with the latest checkpoint, (default False)
            dev_data (machine.dataset.dataset.Dataset, optional): dev Dataset (default None)
            optimizer (machine.optim.Optimizer, optional): optimizer for training
               (default: Optimizer(pytorch.optim.Adam, max_grad_norm=5))
            teacher_forcing_ratio (float, optional): teaching forcing ratio (default 0)
            learing_rate (float, optional): learning rate used by the optimizer (default 0.001)
            checkpoint_path (str, optional): path to load checkpoint from in case training should be resumed
            top_k (int): how many models should be stored during training
        Returns:
            model (machine.models): trained model.
        """
        # If training is set to resume
        if resume:
            resume_checkpoint = Checkpoint.load(checkpoint_path)
            model = resume_checkpoint.model
            self.model = model
            self.optimizer = resume_checkpoint.optimizer

            # A walk around to set optimizing parameters properly
            resume_optim = self.optimizer.optimizer
            defaults = resume_optim.param_groups[0]
            defaults.pop('params', None)
            defaults.pop('initial_lr', None)
            self.optimizer.optimizer = resume_optim.__class__(
                self.model.parameters(), **defaults)

            start_epoch = resume_checkpoint.epoch
            step = resume_checkpoint.step
        else:
            start_epoch = 1
            step = 0
            self.model = model

            def get_optim(optim_name):
                optims = {'adam': optim.Adam, 'adagrad': optim.Adagrad,
                          'adadelta': optim.Adadelta, 'adamax': optim.Adamax,
                          'rmsprop': optim.RMSprop, 'sgd': optim.SGD,
                          None: optim.Adam}
                return optims[optim_name]

            self.optimizer = Optimizer(get_optim(optimizer)(self.model.parameters(),
                                                            lr=learning_rate),
                                       max_grad_norm=5)

        self.logger.info("Optimizer: %s, Scheduler: %s" %
                         (self.optimizer.optimizer, self.optimizer.scheduler))

        # TODO here generate the callbacks based on the available info
        callbacks = CallbackContainer(self,
                                      [Logger(), ModelCheckpoint(top_k=top_k)] + custom_callbacks)

        logs = self._train_epoches(data, num_epochs,
                                   start_epoch, step, dev_data=dev_data,
                                   monitor_data=monitor_data,
                                   callbacks=callbacks,
                                   teacher_forcing_ratio=teacher_forcing_ratio)
        return self.model, logs

    @staticmethod
    def get_batch_data(batch):
        input_variables, input_lengths = getattr(batch, machine.src_field_name)
        target_variables = {'decoder_output': getattr(batch, machine.tgt_field_name),
                            'encoder_input': input_variables}  # The k-grammar metric needs to have access to the inputs

        return input_variables, input_lengths, target_variables

    # @staticmethod
    # def get_losses(losses, metrics, step):
    #     total_loss = 0
    #     model_name = ''
    #     log_msg = ''

    #     for metric in metrics:
    #         val = metric.get_val()
    #         log_msg += '%s %.4f ' % (metric.name, val)
    #         model_name += '%s_%.2f_' % (metric.log_name, val)

    #     for loss in losses:
    #         val = loss.get_loss()
    #         log_msg += '%s %.4f ' % (loss.name, val)
    #         model_name += '%s_%.2f_' % (loss.log_name, val)
    #         total_loss += val

    #     model_name += 's%d' % step

    #     return total_loss, log_msg, model_name
