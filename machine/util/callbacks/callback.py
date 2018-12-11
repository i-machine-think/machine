
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
        log_msg = ''

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
