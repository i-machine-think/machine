from machine.util.callbacks import Callback
import torch


class ReduceLRonPlateauCallback(Callback):
    """
    Callback wrapper for Reduce LR on Plateau
    """

    def __init__(self,
                 monitor='eval_losses',
                 mode='min', factor=0.1,
                 patience=10, verbose=False,
                 threshold=0.0001, threshold_mode='rel',
                 cooldown=0, min_lr=0, eps=1e-08):
        """

        Args:
            monitor (str): Picked from {'eval_losses', 'train_losses'}
            mode (str): One of min, max. In min mode, lr will be reduced when the
                        quantity monitored has stopped decreasing; in max mode
                        it will be reduced when the quantity
                        monitored has stopped increasing. Default: ‘min’.
            factor (float): Factor by which the learning rate will be reduced.
                            new_lr = lr * factor. Default: 0.1.
            patience (int): Number of epochs with no improvement after which
                            learning rate will be reduced. For example,
                            if patience = 2, then we will ignore the first 2 epochs
                            with no improvement, and will only decrease the LR
                            after the 3rd epoch if the loss still hasn’t improved
                            then. Default: 10.
            verbose (bool): If True, prints a message to stdout for each update.
                            Default: False.
            threshold (float): Threshold for measuring the new optimum,
                                to only focus on significant changes.
                                Default: 1e-4.
            threshold_mode (str): One of rel, abs. In rel mode,
                                dynamic_threshold = best * ( 1 + threshold ) in ‘max’ mode
                                or best * ( 1 - threshold ) in min mode.
                                In abs mode, dynamic_threshold = best + threshold in max mode
                                or best - threshold in min mode.
                                Default: ‘rel’.
            cooldown (int): Number of epochs to wait before resuming normal
                            operation after lr has been reduced.
                            Default: 0.
            min_lr (float or list): A scalar or a list of scalars.
                                    A lower bound on the learning rate of
                                    all param groups or each group respectively.
                                    Default: 0.
            eps (float): Minimal decay applied to lr. If the difference between
                        new and old lr is smaller than eps,
                        the update is ignored. Default: 1e-8.
        """
        self.monitor = monitor
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.min_lr = min_lr
        self.eps = eps
        super(ReduceLRonPlateauCallback, self).__init__()

    def on_train_begin(self, info=None):
        # initialize scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.trainer.optimizer.optimizer, mode=self.mode,
            factor=self.factor, patience=self.patience, verbose=self.verbose,
            threshold=self.threshold, threshold_mode=self.threshold_mode,
            min_lr=self.min_lr, eps=self.eps)

        self.trainer.optimizer.set_scheduler(scheduler)

    def on_epoch_end(self, info=None):
        loss_total, _, _ = self.get_losses(
            info[self.monitor], info[self.monitor], info['step'])

        # Updates learning rate using scheduler
        self.trainer.optimizer.update(loss_total, info['epoch'])
