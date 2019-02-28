from machine.util.callbacks import Callback


class EarlyStoppingCallback(Callback):
    """
    Original callback taken from https://github.com/ncullen93/torchsample
    Early Stopping to terminate training early under certain conditions
    """

    def __init__(self,
                 monitor='eval_losses',
                 lm_name=None,
                 min_delta=0,
                 patience=5,
                 minimize=True):
        """
        EarlyStopping callback to exit the training loop if training or
        validation loss or metric does not improve by a certain amount
        for a certain number of epochs
        Args:
            monitor : string in {'eval_losses', 'eval_metrics', 'train_losses', 'train_metrics'}
                whether to monitor train or val loss
            lm_name: loss or metric name eg. 'Avg NLLoss' or 'Word Accuracy'
                    If not specified then the first element
                    in the monitor array is used
                    (default None)
            min_delta : float
                minimum change in monitored value to qualify as improvement.
                This number should be positive.
            patience : integer
                number of epochs to wait for improvment before terminating.
                the counter be reset after each improvment
            minimize: minimize quantity, if false then early stopping will maximize
        """
        if 'loss' in monitor:
            self.loss = True
        elif 'metric' in monitor:
            self.loss = False
        else:
            raise ValueError(
                "Monitor must be string in {'eval_losses', \
                'eval_metrics', 'train_losses', 'train_metrics'}")

        self.monitor = monitor
        self.lm_name = lm_name
        self.minimize = minimize

        self.min_delta = min_delta
        self.patience = patience
        self.wait = 0
        self.best_lm = 1e-15
        super(EarlyStoppingCallback, self).__init__()

    def on_train_begin(self, info=None):
        self.wait = 0
        if self.minimize:
            self.best_lm = 1e15
        else:
            self.best_lm = -1e15

    def on_epoch_end(self, info=None):
        """
        Function called at the end of every epoch
        This allows specifing what eval or train loss to use
        """

        # If specific loss/metric name is specified
        if self.lm_name is not None:
            for lm in info[self.monitor]:
                if lm.name == self.lm_name:
                    current_loss = self.get_loss_metric(lm)
                    break
        else:  # just use the first metric/loss in the array
            current_loss = self.get_loss_metric(info[self.monitor][0])

        # Compare current loss to previous best
        if self.minimize:
            update_best = (current_loss - self.best_lm) < -self.min_delta
        else:
            update_best = (self.best_lm - current_loss) < -self.min_delta

        if update_best:
            self.best_lm = current_loss
            self.wait = 1
        else:
            if self.wait >= self.patience:
                self.trainer._stop_training = True
            self.wait += 1

    def get_loss_metric(self, lm):
        if self.loss:
            return lm.get_loss()
        else:
            return lm.get_val()
