
import logging
import os

from collections import defaultdict
from machine.util.callbacks import Callback


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

    def on_epoch_begin(self, info=None):
        self.logger.info("Epoch: %d, Step: %d" % (info['epoch'], info['step']))

    def on_epoch_end(self, info=None):

        for loss in self.trainer.losses:
            self.epoch_loss_avg[loss.log_name] = \
                self.epoch_loss_total[loss.log_name] \
                / min(info['steps_per_epoch'], info['step'] - info['start_step'])
            self.epoch_loss_total[loss.log_name] = 0

        loss_msg = ' '.join(
            ['%s: %.4f' % (loss.log_name, loss.get_loss()) for loss in self.trainer.losses])

        log_msg = "Finished epoch %d: Train %s" % (info['epoch'], loss_msg)

        if self.trainer.val_data is not None:
            losses, metrics = self.trainer.evaluator.evaluate(
                self.trainer.model, self.trainer.val_data, self.trainer.get_batch_data)
            loss_total, log_, _ = self.get_losses(
                losses, metrics, info['step'])

            # TODO check if this makes sense!
            self.trainer.optimizer.update(loss_total, info['epoch'])
            log_msg += ", Dev set: " + log_
            self.trainer.model.train()
        else:
            # TODO check if this makes sense!
            self.trainer.optimizer.update(self.epoch_loss_avg, info['epoch'])

        self.logger.info(log_msg)

    def on_batch_begin(self, batch, info=None):
        pass

    def on_batch_end(self, batch, info=None):
        # update batch losses
        for loss in self.trainer.losses:
            name = loss.log_name
            self.print_loss_total[name] += loss.get_loss()
            self.epoch_loss_total[name] += loss.get_loss()

        # scheduled printing of losses
        if info['step'] % self.print_every == 0 \
                and info['step_elapsed'] > self.print_every:
            # TODO here all losses should be printed
            # and potentially also computed?
            for loss in self.trainer.losses:
                name = loss.log_name
                self.print_loss_avg[name] = \
                    self.print_loss_total[name] / self.print_every
                self.print_loss_total[name] = 0

            m_logs = {}

            # TODO: Remove evaluate function from here?
            train_losses, train_metrics = self.trainer.evaluator.evaluate(
                self.trainer.model, self.trainer.data, self.trainer.get_batch_data)
            _, train_log_msg, _ = self.get_losses(
                train_losses, train_metrics, info['step'])

            # logs.write_to_log('Train', train_losses,
            #                     train_metrics, step)
            # logs.update_step(step)

            m_logs['Train'] = train_log_msg

            # compute vals for all monitored sets
            # TODO: Remove evaluate function from here?
            for m_data in self.trainer.monitor_data:
                losses, metrics = self.trainer.evaluator.evaluate(
                    self.trainer.model, self.trainer.monitor_data[m_data],
                    self.trainer.get_batch_data)
                _, log_msg, _ = self.get_losses(
                    losses, metrics, info['step'])
                m_logs[m_data] = log_msg
                # logs.write_to_log(m_data, losses, metrics, step)

            all_losses = ' '.join(
                ['%s:\t %s\n' % (os.path.basename(name), m_logs[name]) for name in m_logs])

            log_msg = 'Progress %d%%, %s' % (
                info['step'] / info['total_steps'] * 100,
                all_losses)

            self.logger.info(log_msg)

    def on_train_begin(self, info=None):
        # TODO: somehow fix this by running the initial evaluate
        #      outside the Modelcheckpoint Callback?
        _, log_msg, _ = self.get_losses(
            self.trainer.losses, self.trainer.metrics, info['step'])
        self.logger.info(log_msg)

    def on_train_end(self, info=None):
        pass
