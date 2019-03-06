from machine.util.callbacks import Callback
import numpy as np


class EarlyStoppingCallback(Callback):
    """
    Original callback taken from https://github.com/ncullen93/torchsample
    Early Stopping to terminate training early under certain conditions
    """
    accepted_losses = ['eval_losses', 'train_losses']
    accepted_metrics = ['eval_metrics', 'train_metrics']

    def __init__(self,
                 monitor='eval_losses',
                 objective_name=None,
                 min_delta=0,
                 patience=5,
                 minimize=True):
        """
        EarlyStopping callback to exit the training loop if training or
        validation loss or metric does not improve by a certain amount
        for a certain number of epochs
        Args:
            monitor : string in {'eval_losses',
                'eval_metrics', 'train_losses', 'train_metrics'}
                whether to monitor train or val loss
            objective_name: loss or metric name eg. 'Avg NLLoss' or 'Word Accuracy'
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
        if monitor in self.accepted_losses:
            self.loss = True
        elif monitor in self.accepted_metrics:
            self.loss = False
        else:
            raise ValueError(
                "Monitor must be string in {'eval_losses', \
                'eval_metrics', 'train_losses', 'train_metrics'}")

        self.monitor = monitor
        self.objective_name = objective_name
        self.minimize = 1 if minimize else -1

        self.min_delta = min_delta
        self.patience = patience
        super(EarlyStoppingCallback, self).__init__()

    def on_train_begin(self, info=None):
        self.wait = 0
        self.best = self.minimize * np.Inf

    def on_epoch_end(self, info=None):
        """
        Function called at the end of every epoch
        This allows specifing what eval or train loss to use
        """
        current = self._get_current(info)
        # Compare current loss to previous best
        update_best = self.minimize * \
            (current - self.best) < -self.min_delta

        if update_best:
            self.best = current
            self.wait = 1
        else:
            if self.wait >= self.patience:
                self.trainer._stop_training = True
            self.wait += 1

    def _get_current(self, info):
        """
        Helper function that returns current loss/metric in info
        Uses the objective name stored to find matching metric or loss
        """

        # If specific loss/metric name is specified
        if self.objective_name is not None:
            for objective in info[self.monitor]:
                if objective.name == self.objective_name:
                    return self._get_loss_metric(objective)
        else:  # just use the first metric/loss in the array
            return self._get_loss_metric(info[self.monitor][0])

        raise ValueError('Early Stopping objective_name {} must be present in {}\
                         '.format(self.objective_name,
                                  self.monitor))

    def _get_loss_metric(self, objective):
        return objective.get_loss() if self.loss else objective.get_val()
