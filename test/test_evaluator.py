from __future__ import division
import os
import math
import unittest

from mock import MagicMock, patch, call, ANY
import torchtext

from machine.dataset import SourceField, TargetField
from machine.evaluator import Evaluator
from machine.models.seq2seq import Seq2seq
from machine.models.EncoderRNN import EncoderRNN
from machine.models.DecoderRNN import DecoderRNN
from machine.trainer.supervised_trainer import SupervisedTrainer as trainer


class TestPredictor(unittest.TestCase):

    def setUp(self):
        test_path = os.path.dirname(os.path.realpath(__file__))
        src = SourceField()
        tgt = TargetField()
        self.dataset = torchtext.data.TabularDataset(
            path=os.path.join(test_path, 'data/eng-fra.txt'), format='tsv',
            fields=[('src', src), ('tgt', tgt)],
        )
        src.build_vocab(self.dataset)
        tgt.build_vocab(self.dataset)

        encoder = EncoderRNN(len(src.vocab), 10, 10, 10, rnn_cell='lstm')
        decoder = DecoderRNN(len(tgt.vocab), 10, 10,
                             tgt.sos_id, tgt.eos_id, rnn_cell='lstm')
        self.seq2seq = Seq2seq(encoder, decoder)

        for param in self.seq2seq.parameters():
            param.data.uniform_(-0.08, 0.08)

    @patch.object(Seq2seq, '__call__', return_value=(
        [], None, dict(inputs=[], length=[10] * 64, sequence=MagicMock())))
    @patch.object(Seq2seq, 'eval')
    def test_set_eval_mode(self, mock_eval, mock_call):
        """ Make sure that evaluation is done in evaluation mode. """
        mock_mgr = MagicMock()
        mock_mgr.attach_mock(mock_eval, 'eval')
        mock_mgr.attach_mock(mock_call, 'call')

        evaluator = Evaluator(batch_size=64)
        with patch('machine.evaluator.evaluator.torch.stack', return_value=None), \
                patch('machine.metrics.WordAccuracy.eval_batch', return_value=None), \
                patch('machine.metrics.WordAccuracy.eval_batch', return_value=None), \
                patch('machine.loss.NLLLoss.eval_batch', return_value=None):
            evaluator.evaluate(self.seq2seq, self.dataset,
                               trainer.get_batch_data)

        num_batches = int(math.ceil(len(self.dataset) / evaluator.batch_size))
        expected_calls = [call.eval()] + num_batches * \
            [call.call(ANY, ANY, ANY)]
        self.assertEqual(expected_calls, mock_mgr.mock_calls)
