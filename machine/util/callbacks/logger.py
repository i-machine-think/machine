
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

    def on_epoch_begin(self, info=None):
        self.logger.info("Epoch: %d, Step: %d" % (info['epoch'], info['step']))

    def on_epoch_end(self, info=None):
        for loss in self.trainer.losses:
            self.epoch_loss_avg[loss.log_name] = \
                self.epoch_loss_total[loss.log_name] \
                / max(min(info['steps_per_epoch'], info['step'] - info['start_step']), 1)
            self.epoch_loss_total[loss.log_name] = 0

        if info['step_elapsed'] < 1:
            self.logger.warning("0 Steps elapsed so avg. loss is 0")

        loss_msg = ' '.join(
            ['%s: %.4f' % (loss.log_name, self.epoch_loss_avg[loss.log_name]) for loss in self.trainer.losses])

        _, train_log_msg, _ = self.get_losses(
            info['train_losses'], info['train_metrics'], info['step'])

        log_msg = "Finished epoch %d: Avg Train loss %s, Full Train: %s" % (
            info['epoch'], loss_msg, train_log_msg)

        loss_total, log_, _ = self.get_losses(
            info['eval_losses'], info['eval_metrics'], info['step'])

        # Update learning rate - Needs checking
        self.trainer.optimizer.update(loss_total, info['epoch'])
        log_msg += ", Dev set: " + log_

        self.logger.info(log_msg)

    def on_batch_begin(self, batch, info=None):
        pass

    def on_batch_end(self, batch, info=None):
        # update batch losses
        for loss in self.trainer.batch_losses:
            name = loss.log_name
            self.print_loss_total[name] += loss.get_loss()
            self.epoch_loss_total[name] += loss.get_loss()

        # scheduled printing of losses
        if info['print']:

            # all losses have been already computed and now should be printed

            for loss in self.trainer.batch_losses:
                name = loss.log_name
                self.print_loss_avg[name] = \
                    self.print_loss_total[name] / self.trainer.print_every
                self.print_loss_total[name] = 0

            m_logs = {}

            m_logs['Train Batch'] = ' '.join(
                ['%s %s' % (loss_name, avg_loss) for loss_name, avg_loss in self.print_loss_avg.items()])
            # compute vals for all monitored sets
            for m_data in self.trainer.monitor_data:
                losses = info['monitor_losses'][m_data]
                metrics = info['monitor_metrics'][m_data]
                _, log_msg, _ = self.get_losses(
                    losses, metrics, info['step'])
                m_logs[m_data] = log_msg

            all_losses = ' '.join(
                ['%s:\t %s\n' % (os.path.basename(name), m_logs[name]) for name in m_logs])

            log_msg = 'Progress %d%%, %s' % (
                info['step'] / info['total_steps'] * 100,
                all_losses)

            self.logger.info(log_msg)

    def on_train_begin(self, info=None):
        _, log_msg, _ = self.get_losses(
            info['eval_losses'], info['eval_metrics'], info['step'])
        self.logger.info(log_msg)

    def on_train_end(self, info=None):
        pass
