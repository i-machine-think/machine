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

import seq2seq
from seq2seq.evaluator import Evaluator
from seq2seq.loss import NLLLoss, AttentionLoss
from seq2seq.metrics import WordAccuracy
from seq2seq.optim import Optimizer
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.util.callback import CallbackContainer, Logger, ModelCheckpoint
from seq2seq.util.log import Log

class SupervisedTrainer(object):
    """ The SupervisedTrainer class helps in setting up a training framework in a
    supervised setting.

    Args:
        expt_dir (optional, str): experiment Directory to store details of the experiment,
            by default it makes a folder in the current directory to store the details (default: `experiment`).
        loss (list, optional): list of seq2seq.loss.Loss objects for training (default: [seq2seq.loss.NLLLoss])
        metrics (list, optional): list of seq2seq.metric.metric objects to be computed during evaluation
        batch_size (int, optional): batch size for experiment, (default: 64)
        checkpoint_every (int, optional): number of epochs to checkpoint after, (default: 100)
        print_every (int, optional): number of iterations to print after, (default: 100)
    """
    def __init__(self, expt_dir='experiment', loss=[NLLLoss()], loss_weights=None, metrics=[], batch_size=64, eval_batch_size=128,
                 random_seed=None,
                 checkpoint_every=100, print_every=100):
        self._trainer = "Simple Trainer"
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
        k = NLLLoss()
        self.loss = loss
        self.metrics = metrics
        self.loss_weights = loss_weights or len(loss)*[1.]
        self.evaluator = Evaluator(loss=self.loss, metrics=self.metrics, batch_size=eval_batch_size)
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

    def _train_batch(self, input_variable, input_lengths, target_variable, model, teacher_forcing_ratio):
        loss = self.loss

        # Forward propagation
        decoder_outputs, decoder_hidden, other = model(input_variable, input_lengths, target_variable, teacher_forcing_ratio=teacher_forcing_ratio)

        losses = self.evaluator.compute_batch_loss(decoder_outputs, decoder_hidden, other, target_variable)
        
        # Backward propagation
        for i, loss in enumerate(losses, 0):
            loss.scale_loss(self.loss_weights[i])
            loss.backward(retain_graph=True)
        self.optimizer.step()
        model.zero_grad()

        return losses

    def _train_epoches(self, data, model, n_epochs, start_epoch, start_step,
                       dev_data=None, callbacks=[], monitor_data=[], 
                       teacher_forcing_ratio=0,
                       top_k=5):

        self.set_callbacks(callbacks, top_k=top_k, data=data, dev_data=dev_data)

        iterator_device = torch.cuda.current_device() if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=False, sort_within_batch=True,
            sort_key=lambda x: len(x.src),
            device=iterator_device, repeat=False)

        val_data = dev_data or data

        steps_per_epoch = len(batch_iterator)
        total_steps = steps_per_epoch * n_epochs

        info = dict(zip(['steps_per_epoch', 'total_steps'],\
                [steps_per_epoch, total_steps]))

        info['step'] = start_step
        info['start_epoch'] = start_epoch
        info['epoch'] = start_epoch
        info['start_step'] = start_step
        info['step_elapsed'] = 0

        info['model'] = model       # TODO I find this also a bit hacky

        # store initial model to be sure at least one model is stored
        val_data = dev_data or data

        losses, metrics = self.evaluator.evaluate(model, val_data, self.get_batch_data)

        info['losses'] = losses
        info['metrics'] = metrics

        self.callbacks.on_train_begin(info)

        # TODO this should also be in a callback
        logs = Log()

        for epoch in range(start_epoch, n_epochs + 1):

            self.callbacks.on_epoch_begin(epoch, info)

            batch_generator = batch_iterator.__iter__()

            # consuming seen batches from previous training
            for _ in range((epoch - 1) * steps_per_epoch, info['step']):
                next(batch_generator)

            model.train(True)
            for batch in batch_generator:

                self.callbacks.on_batch_begin(info, batch)
                info['step'] += 1
                info['step_elapsed'] += 1

                input_variables, input_lengths, target_variables = self.get_batch_data(batch)

                losses = self._train_batch(input_variables, input_lengths.tolist(), target_variables, model, teacher_forcing_ratio)

                info['losses'] = losses

                # print log info according to print_every parm
                if info['step'] % self.print_every == 0 and info['step_elapsed'] >= self.print_every:

                    m_logs = {}
                    train_losses, train_metrics = self.evaluator.evaluate(model, data, self.get_batch_data)
                    train_loss, train_log_msg, model_name = self.get_losses(train_losses, train_metrics, info['step'])
                    logs.write_to_log('Train', train_losses, train_metrics, info['step'])
                    logs.update_step(info['step'])

                    m_logs['Train'] = train_log_msg

                    # compute vals for all monitored sets
                    for m_data in monitor_data:
                        m_losses, m_metrics = self.evaluator.evaluate(model, monitor_data[m_data], self.get_batch_data)
                        total_loss, log_msg, model_name = self.get_losses(m_losses, m_metrics, info['step'])
                        m_logs[m_data] = log_msg
                        logs.write_to_log(m_data, m_losses, m_metrics, info['step'])

                    all_losses = ' '.join(['%s:\t %s\n' % (os.path.basename(name), m_logs[name]) for name in m_logs])

                    log_msg = 'Progress %d%%, %s' % (
                            info['step'] / total_steps * 100,
                            all_losses)

                    info['log_msg'] = log_msg

                # check if new model should be saved
                if info['step'] % self.checkpoint_every == 0 or info['step'] == total_steps:
                    # compute dev loss
                    losses, metrics = self.evaluator.evaluate(model, val_data, self.get_batch_data)

                    info['val_losses'] = losses

                self.callbacks.on_batch_end(batch, info)

            if info['step_elapsed'] == 0: continue

            log_msg = ''

            if dev_data is not None:
                losses, metrics = self.evaluator.evaluate(model, dev_data, self.get_batch_data)
                loss_total, log_, model_name = self.get_losses(losses, metrics, info['step'])

                self.optimizer.update(loss_total, epoch)    # TODO check if this makes sense!
                log_msg += ", Dev set: " + log_
                model.train(mode=True)
            else:
                # TODO THIS IS SUPER HACKY, UPDATE IT!!!
                self.optimizer.update(self.callbacks.callbacks[0].epoch_loss_avg, epoch)

            self.callbacks.on_epoch_end(epoch, info)
            info['epoch'] += 1

        self.callbacks.on_train_end(info)

        return logs

    def train(self, model, data, num_epochs=5,
              resume=False, dev_data=None, 
              monitor_data={}, optimizer=None,
              teacher_forcing_ratio=0,
              learning_rate=0.001, checkpoint_path=None, top_k=5):
        """ Run training for a given model.

        Args:
            model (seq2seq.models): model to run training on, if `resume=True`, it would be
               overwritten by the model loaded from the latest checkpoint.
            data (seq2seq.dataset.dataset.Dataset): dataset object to train on
            num_epochs (int, optional): number of epochs to run (default 5)
            resume(bool, optional): resume training with the latest checkpoint, (default False)
            dev_data (seq2seq.dataset.dataset.Dataset, optional): dev Dataset (default None)
            optimizer (seq2seq.optim.Optimizer, optional): optimizer for training
               (default: Optimizer(pytorch.optim.Adam, max_grad_norm=5))
            teacher_forcing_ratio (float, optional): teaching forcing ratio (default 0)
            learing_rate (float, optional): learning rate used by the optimizer (default 0.001)
            checkpoint_path (str, optional): path to load checkpoint from in case training should be resumed
            top_k (int): how many models should be stored during training
        Returns:
            model (seq2seq.models): trained model.
        """
        # If training is set to resume
        if resume:
            resume_checkpoint = Checkpoint.load(checkpoint_path)
            model = resume_checkpoint.model
            self.optimizer = resume_checkpoint.optimizer

            # A walk around to set optimizing parameters properly
            resume_optim = self.optimizer.optimizer
            defaults = resume_optim.param_groups[0]
            defaults.pop('params', None)
            defaults.pop('initial_lr', None)
            self.optimizer.optimizer = resume_optim.__class__(model.parameters(), **defaults)

            start_epoch = resume_checkpoint.epoch
            step = resume_checkpoint.step
        else:
            start_epoch = 1
            step = 0

            def get_optim(optim_name):
                optims = {'adam': optim.Adam, 'adagrad': optim.Adagrad,
                          'adadelta': optim.Adadelta, 'adamax': optim.Adamax,
                          'rmsprop': optim.RMSprop, 'sgd': optim.SGD,
                           None:optim.Adam}
                return optims[optim_name]

            self.optimizer = Optimizer(get_optim(optimizer)(model.parameters(), lr=learning_rate),
                                       max_grad_norm=5)

        self.logger.info("Optimizer: %s, Scheduler: %s" % (self.optimizer.optimizer, self.optimizer.scheduler))

        logs = self._train_epoches(data, model, num_epochs,
                            start_epoch, step, dev_data=dev_data,
                            monitor_data=monitor_data,
                            teacher_forcing_ratio=teacher_forcing_ratio,
                            top_k=top_k)
        return model, logs

    def set_callbacks(self, callbacks, data, dev_data, top_k):
        """
        Create a callback collection and associate it
        with the current trainer.
        """
        # every training outcomes are logged and models
        callbacks = [Logger(), ModelCheckpoint(data, dev_data, top_k=top_k)] + callbacks
        self.callbacks = CallbackContainer(callbacks)
        self.callbacks.set_trainer(self)

    @staticmethod
    def get_batch_data(batch):
        input_variables, input_lengths = getattr(batch, seq2seq.src_field_name)
        target_variables = {'decoder_output': getattr(batch, seq2seq.tgt_field_name),
                            'encoder_input': input_variables}  # The k-grammar metric needs to have access to the inputs

        # If available, also get provided attentive guidance data
        if hasattr(batch, seq2seq.attn_field_name):
            attention_target = getattr(batch, seq2seq.attn_field_name)

            # When we ignore output EOS, the sequence target will not contain the EOS, but if present
            # in the data, the attention indices might still. We should remove this.
            target_length = target_variables['decoder_output'].size(1)
            attn_length = attention_target.size(1)

            # If the attention sequence is exactly 1 longer than the output sequence, the EOS attention
            # index is present.
            if attn_length == target_length + 1:
                # First we replace each of these indices with a -1. This makes sure that the hard
                # attention method will not attend to an input that might not be present (the EOS)
                # We need this if there are attentions of multiple lengths in a bath
                attn_eos_indices = input_lengths.unsqueeze(1) + 1
                attention_target = attention_target.scatter_(dim=1, index=attn_eos_indices, value=-1)
                
                # Next we also make sure that the longest attention sequence in the batch is truncated
                attention_target = attention_target[:, :-1]

            target_variables['attention_target'] = attention_target

        return input_variables, input_lengths, target_variables

    @staticmethod
    def get_losses(losses, metrics, step):
        total_loss = 0
        model_name = ''
        log_msg= ''

        for metric in metrics:
            val = metric.get_val()
            log_msg += '%s %.4f ' % (metric.name, val)
            model_name += '%s_%.2f_' % (metric.log_name, val)

        for loss in losses:
            val = loss.get_loss()
            log_msg += '%s %.4f ' % (loss.name, val)
            model_name += '%s_%.2f_' % (loss.log_name, val)
            total_loss += val

        model_name += 's%d' % step

        return total_loss, log_msg, model_name
