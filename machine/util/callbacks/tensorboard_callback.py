from machine.util.callbacks import Callback


class TensorboardCallback(Callback):
    """
    Callback that is used to store information about
    the training during the training, and writes it to
    a file to the end to be able to read it later.
    """

    def __init__(self, run_path):
        """
        Pass in where to save the run file
        """
        from tensorboardX import SummaryWriter
        super(TensorboardCallback, self).__init__()
        self.writer = SummaryWriter(run_path)

    def on_epoch_end(self, info=None):
        """
        Function called at the end of every epoch
        self.info['train_losses'] and self.info['train_metrics'] should be available to use here.
        self.info['eval_losses'] and self.info['eval_metrics'] should be available to use here.
        """
        for l in info['train_losses']:
            trainl_epoch = ('Train '+l.name+' at Epoch').replace(' ', '_')
            self.writer.add_scalar(trainl_epoch, l.get_loss(), info['epoch'])
        for m in info['train_metrics']:
            trainm_epoch = ('Train ' + m.name+' at Epoch').replace(' ', '_')
            self.writer.add_scalar(trainm_epoch, m.get_val(), info['epoch'])

        for l in info['eval_losses']:
            evall_epoch = ('Valid '+l.name+' at Epoch').replace(' ', '_')
            self.writer.add_scalar(evall_epoch, l.get_loss(), info['epoch'])
        for m in info['eval_metrics']:
            evalm_epoch = ('Valid '+m.name+' at Epoch').replace(' ', '_')
            self.writer.add_scalar(evalm_epoch, m.get_val(), info['epoch'])

    def on_batch_end(self, batch, info=None):
        """
        Function called at the end of every batch
        If self.info['print'] = True:
            Then self.info['monitor_losses'] and self.info['monitor_metrics']
            should be available to use here.
        If self.info['checkpoint'] = True:,
            Then self.info['eval_losses'] and self.info['eval_metrics']
            should be available to use here.
        """
        # Track Validation Loss and metrics
        if info['checkpoint']:
            for l in info['eval_losses']:
                evall_step = ('Valid '+l.name+' at Step').replace(' ', '_')
                self.writer.add_scalar(
                    evall_step, l.get_loss(), info['step'])
            for m in info['eval_metrics']:
                evalm_step = ('Valid '+m.name+' at Step').replace(' ', '_')
                self.writer.add_scalar(evalm_step, m.get_val(), info['step'])

        if info['print']:
            for monitor in info['monitor_losses']:
                for l in info['monitor_losses'][monitor]:
                    testl_step = (monitor+' '+l.name +
                                  ' at Step').replace(' ', '_')
                    self.writer.add_scalar(
                        testl_step, l.get_loss(), info['step'])
                for m in info['monitor_metrics'][monitor]:
                    testm_step = (monitor+' '+m.name +
                                  ' at Step').replace(' ', '_')
                    self.writer.add_scalar(
                        testm_step, m.get_val(), info['step'])

    def on_train_begin(self, info=None):
        """
        Function called at the very beginning of every training
        self.info['eval_losses'] and self.info['eval_metrics']
        should be available to use here.
        """
        for l in info['eval_losses']:
            evall_step = ('Valid '+l.name+' at Step').replace(' ', '_')
            self.writer.add_scalar(
                evall_step, l.get_loss(), info['step'])
        for m in info['eval_metrics']:
            evalm_step = ('Valid '+m.name+' at Step').replace(' ', '_')
            self.writer.add_scalar(evalm_step, m.get_val(), info['step'])
