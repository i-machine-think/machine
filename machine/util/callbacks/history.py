from machine.util.callbacks import Callback
from machine.util import Log


class History(Callback):
    """
    Callback that is used to store information about
    the training during the training, and writes it to
    a file to the end to be able to read it later.
    """

    def __init__(self):
        super(History, self).__init__()
        self.logs = Log()

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_epoch_begin(self, info=None):
        pass

    def on_epoch_end(self, info=None):
        # self.logs.write_to_log('Train', info['train_losses'],
        #                            info['train_metrics'], info['step'])
        pass

    def on_batch_begin(self, batch, info=None):
        pass

    def on_batch_end(self, batch, info=None):
        if info['print']:
            self.logs.update_step(info['step'])
            for m_data in self.trainer.monitor_data:
                self.logs.write_to_log(m_data,
                                       info['monitor_losses'][m_data],
                                       info['monitor_metrics'][m_data],
                                       info['step'])

    def on_train_begin(self, info=None):
        pass

    def on_train_end(self, info=None):
        pass
