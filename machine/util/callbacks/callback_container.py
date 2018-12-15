class CallbackContainer(object):

    def __init__(self, trainer, callbacks=[]):
        self.callbacks = callbacks
        self.info = {}
        self.set_trainer(trainer)

    def set_trainer(self, trainer):
        self.trainer = trainer
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def set_info(self, start_step, start_epoch,
                 steps_per_epoch, total_steps):
        self.info['start_step'] = start_step
        self.info['step'] = start_step
        self.info['start_epoch'] = start_epoch
        self.info['epoch'] = start_epoch
        self.info['step_elapsed'] = 0
        self.info['steps_per_epoch'] = steps_per_epoch
        self.info['total_steps'] = total_steps
        self.info['print'] = False
        self.info['checkpoint'] = False

    def on_epoch_begin(self, epoch):
        self.info['epoch'] = epoch
        for callback in self.callbacks:
            callback.on_epoch_begin(self.info)

    def on_epoch_end(self, epoch):
        self.info['epoch'] = epoch

        self.info['eval_losses'], self.info['eval_metrics'] \
            = self._evaluate_model_on_validation()

        for callback in self.callbacks:
            callback.on_epoch_end(self.info)

    def on_batch_begin(self, batch):
        for callback in self.callbacks:
            callback.on_batch_begin(batch, self.info)

        self.info['step'] += 1
        self.info['step_elapsed'] += 1

    def on_batch_end(self, batch):

        # Necessary check for logger (done here so as to pass
        # losses + metrics to info and allow other callbacks to
        # access the information
        if self.info['step'] % self.trainer.print_every == 0 \
                and self.info['step_elapsed'] > self.trainer.print_every:
            self.info['print'] = True

            # If we 'print' then we want to log, therefore we evaluate
            # for loss on whole train and on all monitored datasets
            self.info['train_losses'], self.info['train_metrics'] = self.trainer.evaluator.evaluate(
                self.trainer.model, self.trainer.train_data, self.trainer.get_batch_data)

            # Evaluate on all Monitor datasets
            self.info['monitor_losses'] = {}
            self.info['monitor_metrics'] = {}
            for m_data in self.trainer.monitor_data:
                self.info['monitor_losses'][m_data], self.info['monitor_metrics'][m_data] \
                    = self.trainer.evaluator.evaluate(self.trainer.model,
                                                      self.trainer.monitor_data[m_data],
                                                      self.trainer.get_batch_data)

        # If checkpoint time then evaluate on validation set
        # Again this is done here in order for other callbacks
        # to have access to the evaluation results
        if self.info['step'] % self.trainer.checkpoint_every == 0 or \
                self.info['step'] == self.info['total_steps']:
            self.info['checkpoint'] = True
            self.info['eval_losses'], self.info['eval_metrics'] \
                = self._evaluate_model_on_validation()

        for callback in self.callbacks:
            callback.on_batch_end(batch, self.info)

        self.info['print'] = False
        self.info['checkpoint'] = False

    def on_train_begin(self):

        # set the best eval losses based on starting accuracies
        self.info['eval_losses'], self.info['eval_metrics'] \
            = self._evaluate_model_on_validation()

        for callback in self.callbacks:
            callback.on_train_begin(self.info)

    def on_train_end(self):
        logs = {}
        for callback in self.callbacks:
            callback.on_train_end(self.info)

            # Gets log object from History call back
            if getattr(callback, logs):
                logs = callback.logs
        return logs

    def _evaluate_model_on_validation(self):
        return self.trainer.evaluator.evaluate(self.trainer.model,
                                               self.trainer.val_data,
                                               self.trainer.get_batch_data)
