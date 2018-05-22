import math
import random
import unittest

import torch
import torch.nn.functional as F

from seq2seq.loss.loss import Loss
from seq2seq.loss import NLLLoss, Perplexity

class TestLoss(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        num_class = 5
        batch_size = 5
        num_batch = 10
        cls.num_batch = num_batch
        cls.outputs = [F.softmax(torch.randn(batch_size, num_class), dim=1)
                   for _ in range(num_batch)]
        cls.targets = [torch.LongTensor([random.randint(0, num_class - 1)
                                              for _ in range(batch_size)])
                   for _ in range(num_batch)]

    def test_loss_init(self):
        name = "name"
        shortname = "shortname"
        inputs="decoder_output"
        targets="decoder_output"
        loss = Loss(name, shortname, inputs, targets, torch.nn.NLLLoss())
        self.assertEqual(loss.name, name)
        self.assertEqual(loss.log_name, shortname)

    def test_loss_init_WITH_NON_LOSS(self):
        self.assertRaises(ValueError, lambda: Loss("name", "shortname", "decoder_outputs", "decoder_targets", "loss"))

    def test_loss_backward_WITH_NO_LOSS(self):
        loss = Loss("name", "shortname", "decoder_output", "decoder_output", torch.nn.NLLLoss())
        self.assertRaises(ValueError, lambda: loss.backward())

    def test_nllloss_init(self):
        loss = NLLLoss()
        self.assertEqual(loss.name, NLLLoss._NAME)
        self.assertEqual(loss.log_name, NLLLoss._SHORTNAME)
        self.assertTrue(type(loss.criterion) is torch.nn.NLLLoss)

    def test_nllloss(self):
        loss = NLLLoss()
        pytorch_loss = 0
        pytorch_criterion = torch.nn.NLLLoss()
        for output, target in zip(self.outputs, self.targets):
            loss.eval_step(output, target)
            pytorch_loss += pytorch_criterion(output, target)

        loss_val = loss.get_loss()
        pytorch_loss /= self.num_batch

        self.assertAlmostEqual(loss_val, pytorch_loss.item())

    def test_nllloss_WITH_OUT_SIZE_AVERAGE(self):
        loss = NLLLoss(size_average=False)
        pytorch_loss = 0
        pytorch_criterion = torch.nn.NLLLoss(size_average=False)
        for output, target in zip(self.outputs, self.targets):
            loss.eval_step(output, target)
            pytorch_loss += pytorch_criterion(output, target)

        loss_val = loss.get_loss()

        self.assertAlmostEqual(loss_val, pytorch_loss.item())

    def test_perplexity_init(self):
        loss = Perplexity()
        self.assertEqual(loss.name, Perplexity._NAME)
        self.assertEqual(loss.log_name, Perplexity._SHORTNAME)

    def test_perplexity(self):
        nll = NLLLoss()
        ppl = Perplexity()
        for output, target in zip(self.outputs, self.targets):
            nll.eval_step(output, target)
            ppl.eval_step(output, target)

        nll_loss = nll.get_loss()
        ppl_loss = ppl.get_loss()

        self.assertAlmostEqual(ppl_loss, math.exp(nll_loss))
