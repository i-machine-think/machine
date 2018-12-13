class CallbackContainer(object):

    def __init__(self, trainer, callbacks=[]):
        self.callbacks = callbacks
        self.info = {}
        self.set_trainer(trainer)

    def set_trainer(self, trainer):
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

    def on_epoch_begin(self, epoch):
        self.info['epoch'] = epoch
        for callback in self.callbacks:
            callback.on_epoch_begin(self.info)

    def on_epoch_end(self, epoch):
        self.info['epoch'] = epoch
        for callback in self.callbacks:
            callback.on_epoch_end(self.info)

    def on_batch_begin(self, batch):
        for callback in self.callbacks:
            callback.on_batch_begin(batch, self.info)

        self.info['step'] += 1
        self.info['step_elapsed'] += 1

    def on_batch_end(self, batch):
        for callback in self.callbacks:
            callback.on_batch_end(batch, self.info)

    def on_train_begin(self):
        for callback in self.callbacks:
            callback.on_train_begin(self.info)

    def on_train_end(self):
        for callback in self.callbacks:
            callback.on_train_end(self.info)
