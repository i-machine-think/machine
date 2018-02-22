from __future__ import print_function, division

import torch
import torchtext

import seq2seq
from seq2seq.loss import NLLLoss

class Evaluator(object):
    """ Class to evaluate models with given datasets.

    Args:
        loss (seq2seq.loss, optional): loss for evaluator (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for evaluator (default: 64)
    """

    def __init__(self, loss=[NLLLoss()], metrics=['word_accuracy', 'sequence_accuracy'], batch_size=64):
        self.losses = loss
        self.metrics = metrics
        self.batch_size = batch_size

    def update_batch_metrics(self, metrics, other, target_variable):

        outputs = other['sequence']

        for metric in metrics:
            metric.eval_batch(outputs, target_variable)

        return metrics

    def compute_batch_loss(self, decoder_outputs, decoder_hidden, other, target_variable):

        losses = self.losses
        for loss in losses:
            loss.reset()

        losses = self.update_loss(losses, decoder_outputs, decoder_hidden, other, target_variable)

        return losses

    def update_loss(self, losses, decoder_outputs, decoder_hidden, other, target_variable):

        batch_size = target_variable.size(0)
        for step, step_output in enumerate(decoder_outputs):
            target = target_variable[:, step + 1]
            for loss in losses:
                loss.eval_batch(step_output.contiguous().view(batch_size, -1), target)

        return losses

    def evaluate(self, model, data):
        """ Evaluate a model on given dataset and return performance.

        Args:
            model (seq2seq.models): model to evaluate
            data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against

        Returns:
            loss (float): loss of the given model on the given dataset
            accuracy (float): accuracy of the given model on the given dataset
        """
        model.eval()

        losses = self.losses
        for loss in losses:
            loss.reset()

        metrics = self.metrics
        for metric in metrics:
            metric.reset()

        # create batch iterator
        device = None if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=True, sort_key=lambda x: len(x.src),
            device=device, train=False)

        # loop over batches
        for batch in batch_iterator:
            input_variable, input_lengths  = getattr(batch, seq2seq.src_field_name)
            target_variable = getattr(batch, seq2seq.tgt_field_name)

            decoder_outputs, decoder_hidden, other = model(input_variable, input_lengths.tolist(), target_variable)

            losses = self.update_loss(losses, decoder_outputs, decoder_hidden, other, target_variable)

            metrics = self.update_batch_metrics(metrics, other, target_variable)

        accuracy = metrics[0].get_val()
        seq_accuracy = metrics[1].get_val()

        return losses, metrics
