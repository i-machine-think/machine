import logging
from collections import defaultdict
from machine.util.callbacks import Callback


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

        # if path is not None:
        #     self.read_from_file(path)

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_epoch_begin(self, info=None):
        pass

    def on_epoch_end(self, info=None):
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
