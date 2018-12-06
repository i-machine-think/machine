from __future__ import print_function
import logging
import os
import shutil

import seq2seq
from seq2seq.util.checkpoint import Checkpoint

from collections import defaultdict

class CallbackContainer(object):

    def __init__(self, callbacks=[]):
        self.callbacks = callbacks

    def set_trainer(self, trainer):
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def set_params(self, params):
        for callback in self.callbacks:
            self.params = params

    def on_epoch_begin(self, epoch, info=None):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, info)

    def on_epoch_end(self, epoch, info=None):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, info)

    def on_batch_begin(self, batch, info=None):
        for callback in self.callbacks:
            callback.on_batch_begin(batch, info)

    def on_batch_end(self, batch, info=None):
        for callback in self.callbacks:
            callback.on_batch_end(batch, info)

    def on_train_begin(self, info=None):
        for callback in self.callbacks:
            callback.on_train_begin(info)

    def on_train_end(self, info=None):
        for callback in self.callbacks:
            callback.on_train_end(info)

class Callback(object):
    """
    Abstract base class to build callbacks.

    Inspired by keras' callbacks.
    A callback is a set of functions to be applied at given stages
    of the training procedure. You can use callbacks to get a view
    on internal states and statistics of the model during training. 
    You can pass a list of callbacks (as the keyword argument callbacks)
    to the train() method of the SupervisedTrainer. 
    The relevant methods of the callbacks will then be called at each 
    stage of the training.
    """

    def __init__(self):
        pass

    def set_params(self, params):
        self.params = params

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_epoch_begin(self, epoch, info=None):
        pass

    def on_epoch_end(self, epoch, info=None):
        pass

    def on_batch_begin(self, batch, info=None):
        pass

    def on_batch_end(self, batch, info=None):
        pass

    def on_train_begin(self, info=None):
        pass

    def on_train_end(self, info=None):
        pass

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




class History(Callback):
    """
    Callback that is used to store information about
    the training during the training, and writes it to
    a file to the end to be able to read it later.
    """

    def __init__(self):
        super(History, self).__init__()
        self.steps = []
        self.log = defaultdict(lambda: defaultdict(list))

        self.logging = logging.getLogger(__name__)

        if path is not None:
            self.read_from_file(path)

    def set_params(self, params):
        self.params = params

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_epoch_begin(self, epoch, info=None):
        pass

    def on_epoch_end(self, epoch, info=None):
        pass

    def on_batch_begin(self, batch, info=None):
        pass

    def on_batch_end(self, batch, info=None):
        pass

    def on_train_begin(self, info=None):
        pass

    def on_train_end(self, info=None):
        pass

    def write_to_log(self, dataname, losses, metrics, step):
        """
        Add new losses to Log object.
        """
        for metric in metrics:
            val = metric.get_val()
            self.log[dataname][metric.log_name].append(val)

        for loss in losses:
            val = loss.get_loss()
            self.log[dataname][loss.log_name].append(val)


class Logger(Callback):
    """
    Callback that is used to log information during
    training
    """

    def __init__(self):
        super(Logger, self).__init__()

        self.logger = logging.getLogger(__name__)
        self.print_loss_total = defaultdict(float)  # Reset every print_every
        self.epoch_loss_total = defaultdict(float)  # Reset every epoch
        self.epoch_loss_avg = defaultdict(float)
        self.print_loss_avg = defaultdict(float)

    def set_trainer(self, trainer):
        self.trainer = trainer
        self.print_every = self.trainer.print_every

    def on_epoch_begin(self, epoch, info=None):
        step = info['step']
        self.logger.info("Epoch: %d, Step: %d" % (epoch, step))

    def on_epoch_end(self, epoch, info=None):
        for loss in info['losses']:
            self.epoch_loss_avg[loss.log_name] = \
                    self.epoch_loss_total[loss.log_name] \
                    / min(info['steps_per_epoch'], info['step'] \
                    - info['start_step'])
            self.epoch_loss_total[loss.log_name] = 0

        loss_msg = ' '.join(['%s: %.4f' % (loss.log_name, loss.get_loss()) for loss in info['losses']])

        log_msg = "Finished epoch %d: Train %s" % (epoch, loss_msg)

        self.logger.info(log_msg)

    def on_batch_begin(self, batch, info=None):
        pass

    def on_batch_end(self, batch, info=None):
        # update batch losses
        for loss in info['losses']:
            name = loss.log_name
            self.print_loss_total[name] += loss.get_loss()
            self.epoch_loss_total[name] += loss.get_loss()

        # scheduled printing of losses
        if info['step'] % self.print_every == 0 \
                and info['step_elapsed'] > self.print_every:
            # TODO here all losses should be printed
            # and potentially also computed?
            for loss in info['losses']:
                name = loss.log_name
                self.print_loss_avg[name] = \
                        self.print_loss_total[name] / self.print_every
                self.print_loss_total[name] = 0
            
            self.logger.info(info['log_msg'])


    def on_train_begin(self, info=None):
        losses = info['losses']
        metrics = info['metrics']
        step = info['step']
        total_loss, log_msg, model_name = self.get_losses(losses, metrics, step)
        self.logger.info(log_msg)

    def on_train_end(self, info=None):
        pass

class ModelCheckpoint(Callback):
    """
    Model checkpoint to save weights during training. 
    This callback is automatically applied for every model that
    is trained with the SupervisedTrainer.
    """

    def __init__(self, data, dev_data, top_k=5, monitor='val', 
            save_best_only=True):
        super(ModelCheckpoint, self).__init__()
        self.top_k = top_k
        self.save_best_only = save_best_only
        self.data = data
        self.dev_data = dev_data

    def set_params(self, params):
        self.params = params

    def set_trainer(self, trainer):
        self.trainer = trainer
        self.checkpoint_every = trainer.checkpoint_every
        self.expt_dir = trainer.expt_dir

    def on_epoch_begin(self, epoch, info=None):
        pass

    def on_epoch_end(self, epoch, info=None):
        pass

    def on_batch_begin(self, batch, info=None):
        pass

    def on_batch_end(self, batch, info=None):

        # this check is also hacky, occurs also in
        # supervised trainer to not compute the dev
        # loss too often
        if info['step'] % self.checkpoint_every == 0 or \
                info['step'] == info['total_steps']:
            total_loss, log_msg, model_name = \
                    self.get_losses(info['losses'], info['metrics'], info['step'])

            max_eval_loss = max(self.loss_best)

            if total_loss < max_eval_loss:
                    index_max = self.loss_best.index(max_eval_loss)
                    # rm prev model
                    if self.best_checkpoints[index_max] is not None:
                        shutil.rmtree(os.path.join(self.expt_dir, self.best_checkpoints[index_max]))
                    self.best_checkpoints[index_max] = model_name
                    self.loss_best[index_max] = total_loss

                    # save model
                    Checkpoint(model=info['model'],
                               optimizer=self.trainer.optimizer,
                               epoch=info['epoch'], step=info['step'],
                               input_vocab=self.data.fields[seq2seq.src_field_name].vocab,
                               output_vocab=self.data.fields[seq2seq.tgt_field_name].vocab).save(self.expt_dir, name=model_name)

    def on_train_begin(self, info=None):

        # set the best losses based on starting accuracies

        total_loss, log_msg, model_name = \
                self.get_losses(info['losses'], info['metrics'], info['step'])

        self.loss_best = self.top_k*[total_loss]
        self.best_checkpoints = self.top_k*[None]
        self.best_checkpoints[0] = model_name

        # store first model

        Checkpoint(model=info['model'],
                   optimizer=self.trainer.optimizer,
                   epoch=info['start_epoch'], step=info['start_step'],
                   input_vocab=self.data.fields[seq2seq.src_field_name].vocab,
                   output_vocab=self.data.fields[seq2seq.tgt_field_name].vocab).save(self.expt_dir, name=model_name)

    def on_train_end(self, info=None):
        # TODO perhaps here also the model should be saved?
        pass

    def save(self, name):
        """
        Saves the current model and related training parameters into a
        subdirectory of the directory stored in self.experiment_dir.

        name (str): alternative name for the model

        Returns:
             str: path to the saved checkpoint subdirectory
        """

        self._path = os.path.join(experiment_dir, name)
        path = self._path

        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        torch.save({'epoch': self.epoch,
                    'step': self.step,
                    'optimizer': self.optimizer
                   },
                   os.path.join(path, self.TRAINER_STATE_NAME))
        torch.save(self.model, os.path.join(path, self.MODEL_NAME))

        with open(os.path.join(path, self.INPUT_VOCAB_FILE), 'wb') as fout:
            dill.dump(self.input_vocab, fout)
        with open(os.path.join(path, self.OUTPUT_VOCAB_FILE), 'wb') as fout:
            dill.dump(self.output_vocab, fout)

        return path
