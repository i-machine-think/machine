from __future__ import division
import logging
import os
import random
import shutil

import torch
import torchtext
from torch import optim

from collections import defaultdict

import machine
from machine.evaluator import Evaluator
from machine.loss import NLLLoss
from machine.optim import Optimizer
from machine.util.checkpoint import Checkpoint
from machine.util.callbacks import CallbackContainer, Logger, ModelCheckpoint, History
from machine.util.log import Log

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SupervisedTrainer(object):
    """ The SupervisedTrainer class helps in setting up a training framework in a
    supervised setting.

    Args:
        expt_dir (optional, str): experiment Directory to store details of the experiment,
            by default it makes a folder in the current directory to store the details (default: `experiment`).
    """

    def __init__(self, expt_dir='experiment'):
        self._trainer = "Simple Trainer"

        if not os.path.isabs(expt_dir):
            expt_dir = os.path.join(os.getcwd(), expt_dir)
        self.expt_dir = expt_dir
        if not os.path.exists(self.expt_dir):
            os.makedirs(self.expt_dir)

    def set_local_parameters(self, random_seed, losses, metrics,
                             loss_weights, checkpoint_every, print_every):
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)

        self.losses = losses
        self.metrics = metrics
        self.loss_weights = loss_weights or len(losses)*[1.]
        self.evaluator = Evaluator(loss=self.losses, metrics=self.metrics)
        self.optimizer = None
        self.checkpoint_every = checkpoint_every
        self.print_every = print_every
        self.logger = logging.getLogger(__name__)
        self._stop_training = False

    def _train_batch(self, input_variable, input_lengths, target_variable, teacher_forcing_ratio):
        loss = self.losses

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
                       dev_data, monitor_data=[],
                       teacher_forcing_ratio=0):

        steps_per_epoch = len(data)
        total_steps = steps_per_epoch * n_epochs

        # give start information to callbacks
        callbacks.set_info(start_step, start_epoch,
                           steps_per_epoch,
                           total_steps)

        # set data as attribute to trainer
        self.train_data = data
        self.val_data = dev_data
        self.monitor_data = monitor_data

        # ########################################
        # This is used to resume training from same place in dataset
        # after loading from checkpoint
        s = start_step
        if start_epoch > 1:
            s -= (start_epoch - 1) * steps_per_epoch

        ########################################

        # Call all callbacks
        callbacks.on_train_begin()
        for epoch in range(start_epoch, n_epochs + 1):

            callbacks.on_epoch_begin(epoch)

            self.model.train()
            for batch in data:

                # Skip over the batches that are below start step
                if epoch == start_epoch and s > 0:
                    s -= 1
                    continue

                callbacks.on_batch_begin(batch)

                input_variables, input_lengths, target_variables = self.get_batch_data(
                    batch)

                self.batch_losses = self._train_batch(input_variables,
                                                      input_lengths,
                                                      target_variables,
                                                      teacher_forcing_ratio)
                callbacks.on_batch_end(batch)

            callbacks.on_epoch_end(epoch)

            # Stop training early if flag _stop_training is True
            if self._stop_training:
                break

        logs = callbacks.on_train_end()
        return logs

    def train(self, model, data,
              dev_data,
              num_epochs=5,
              resume_training=False,
              monitor_data={},
              optimizer=None,
              teacher_forcing_ratio=0,
              custom_callbacks=[],
              learning_rate=0.001,
              checkpoint_path=None,
              top_k=5,
              losses=[NLLLoss()],
              loss_weights=None,
              metrics=[],
              random_seed=None,
              checkpoint_every=100,
              print_every=100):
        """ Run training for a given model.

        Args:
            model (machine.models): model to run training on, if `resume=True`, it would be
               overwritten by the model loaded from the latest checkpoint.
            data (torchtext.data.Iterator: torchtext iterator object to train on
            num_epochs (int, optional): number of epochs to run (default 5)
            resume_training(bool, optional): resume training with the latest checkpoint up until the number of epochs (default False)
            dev_data (torchtext.data.Iterator): dev/validation set iterator
                Note: must not pass in the train iterator here as this gets evaluated during training (in between batches)
                If you want to evaluate on the full train during training then make two iterators and pass the second one here
            monitor_data (list of torchtext.data.Iterator, optional): list of iterators to test on (default None)
                Note: must not pass in the train iterator here as this gets evaluated during training (in between batches)
                      If you want to evaluate on the full train during training then make two iterators and pass the second one here
            optimizer (machine.optim.Optimizer, optional): optimizer for training
               (default: Optimizer(pytorch.optim.Adam, max_grad_norm=5))
            teacher_forcing_ratio (float, optional): teaching forcing ratio (default 0)
            custom_callbacks (list, optional): list of custom call backs (see utils.callbacks.callback for base class)
            learing_rate (float, optional): learning rate used by the optimizer (default 0.001)
            checkpoint_path (str, optional): path to load checkpoint from in case training should be resumed
            top_k (int): how many models should be stored during training
            loss (list, optional): list of machine.loss.Loss objects for training (default: [machine.loss.NLLLoss])
            metrics (list, optional): list of machine.metric.metric objects to be computed during evaluation
            checkpoint_every (int, optional): number of epochs to checkpoint after, (default: 100)
            print_every (int, optional): number of iterations to print after, (default: 100)
        Returns:
            model (machine.models): trained model.
        """
        self.set_local_parameters(random_seed, losses, metrics,
                                  loss_weights, checkpoint_every, print_every)
        # If training is set to resume
        if resume_training:
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

        callbacks = CallbackContainer(self,
                                      [Logger(),
                                       ModelCheckpoint(top_k=top_k),
                                       History()] + custom_callbacks)

        logs = self._train_epoches(data, num_epochs,
                                   start_epoch, step, dev_data=dev_data,
                                   monitor_data=monitor_data,
                                   callbacks=callbacks,
                                   teacher_forcing_ratio=teacher_forcing_ratio)

        return self.model, logs

    @staticmethod
    def get_batch_data(batch):
        # TODO Maybe move this method / or make optional - this is seq2seq specific
        input_variables, input_lengths = getattr(batch, machine.src_field_name)
        target_variables = {'decoder_output': getattr(batch, machine.tgt_field_name),
                            'encoder_input': input_variables}  # The k-grammar metric needs to have access to the inputs

        return input_variables, input_lengths, target_variables
