import unittest
import os

import mock
import torchtext

from machine.dataset import SourceField, TargetField
from machine.trainer import SupervisedTrainer
from machine.util.callbacks import CallbackContainer
from machine.evaluator import Evaluator


class TestSupervisedTrainer(unittest.TestCase):

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

        self.data_iterator = torchtext.data.BucketIterator(
            dataset=self.dataset, batch_size=4,
            sort=False, sort_within_batch=True,
            sort_key=lambda x: len(x.src),
            repeat=False)

    @mock.patch('machine.trainer.SupervisedTrainer._train_batch',
                return_value=[])
    @mock.patch('machine.util.checkpoint.Checkpoint.save')
    @mock.patch('machine.evaluator.Evaluator.evaluate', return_value=([], []))
    def test_batch_num_when_resuming(self, mock_evaluator, mock_checkpoint, mock_func):

        trainer = SupervisedTrainer()
        trainer.model = mock.Mock()
        trainer.optimizer = mock.Mock()

        callbacks = CallbackContainer(trainer)

        n_epoches = 1
        start_epoch = 1
        steps_per_epoch = len(self.data_iterator)
        step = 3
        trainer.set_local_parameters(123, [], [], [], 1000, 1000)
        trainer._train_epoches(self.data_iterator, n_epoches,
                               start_epoch, step, callbacks)
        print(mock_func)
        self.assertEqual(steps_per_epoch - step, mock_func.call_count)

    @mock.patch('machine.trainer.SupervisedTrainer._train_batch',
                return_value=0)
    @mock.patch('machine.util.checkpoint.Checkpoint.save')
    @mock.patch('machine.evaluator.Evaluator.evaluate', return_value=([], []))
    def test_resume_from_multiple_of_epoches(self, mock_evaluator, mock_checkpoint, mock_func):
        mock_optim = mock.Mock()

        trainer = SupervisedTrainer()
        trainer.model = mock.Mock()
        trainer.optimizer = mock.Mock()

        callbacks = CallbackContainer(trainer)

        n_epoches = 1
        start_epoch = 1
        step = 7
        trainer.set_local_parameters(123, [], [], [], 1000, 1000)
        trainer._train_epoches(
            self.data_iterator, n_epoches, start_epoch, step, callbacks)

    @mock.patch('machine.util.checkpoint.Checkpoint')
    @mock.patch('machine.util.checkpoint.Checkpoint.load')
    @mock.patch('machine.optim.Optimizer')
    @mock.patch('torch.optim.SGD')
    @mock.patch('machine.trainer.SupervisedTrainer._train_epoches')
    def test_loading_optimizer(
            self, train_func, sgd, optimizer, load_function, checkpoint):

        load_function.returnvalue = checkpoint
        mock_model = mock.Mock()
        mock_model.params.returnvalue = True
        n_epoches = 2

        trainer = SupervisedTrainer()

        trainer.train(mock_model, self.data_iterator, n_epoches,
                      resume=True, checkpoint_path='dummy', optimizer='sgd')

        self.assertFalse(
            sgd.called, "Failed to not call Optimizer() when optimizer should be loaded from checkpoint")

        trainer.train(mock_model, self.data_iterator, n_epoches,
                      resume=False, checkpoint_path='dummy', optimizer='sgd')

        sgd.assert_called()

        return


if __name__ == '__main__':
    unittest.main()
