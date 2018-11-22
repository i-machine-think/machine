from __future__ import print_function
import math
import torch.nn as nn
import torch
import numpy as np

class Loss(object):
    """ Base class for encapsulation of the loss functions.

    This class defines interfaces that are commonly used with loss functions
    in training and inferencing.  For information regarding individual loss
    functions, please refer to http://pytorch.org/docs/master/nn.html#loss-functions

    Note:
        Do not use this class directly, use one of the sub classes.

    Args:
        name (str): name of the loss function used by logging messages.
        criterion (torch.nn._Loss): one of PyTorch's loss functions.  Refer
            to http://pytorch.org/docs/master/nn.html#loss-functions for
            a list of them.

    Attributes:
        name (str): name of the loss function used by logging messages.
        criterion (torch.nn._Loss): one of PyTorch's loss functions.  Refer
            to http://pytorch.org/docs/master/nn.html#loss-functions for
            a list of them.  Implementation depends on individual
            sub-classes.
        acc_loss (int or torcn.nn.Tensor): variable that stores accumulated loss.
        norm_term (float): normalization term that can be used to calculate
            the loss of multiple batches.  Implementation depends on individual
            sub-classes.
    """

    def __init__(self, name, log_name, inputs, target, criterion):
        self.name = name
        self.log_name = log_name
        self.inputs = inputs
        self.target = target
        self.criterion = criterion
        if not issubclass(type(self.criterion), nn.modules.loss._Loss):
            raise ValueError("Criterion has to be a subclass of torch.nn._Loss")
        # accumulated loss
        self.acc_loss = 0
        # normalization term
        self.norm_term = 0

    def reset(self):
        """ Reset the accumulated loss. """
        self.acc_loss = 0
        self.norm_term = 0

    def get_loss(self):
        """ Get the loss.

        This method defines how to calculate the averaged loss given the
        accumulated loss and the normalization term.  Override to define your
        own logic.

        Returns:
            loss (float): value of the loss.
        """
        raise NotImplementedError

    def eval_batch(self, decoder_outputs, other, target_variable):
        """ Evaluate and accumulate loss given outputs and expected results.

        This method is called after each batch with the batch outputs and
        the target (expected) results.  The loss and normalization term are
        accumulated in this method.  Override it to define your own accumulation
        method.

        Args:
            decoder_outputs (torch.Tensor): outputs of a batch.
            other (dictionary): extra outputs of the model
            target_variable (torch.Tensor): expected output of a batch.
        """

        # lists with:
        # decoder outputs # (batch, vocab_size?)
        # attention scores # (batch, 1, input_length)

        if self.inputs == 'decoder_output':
            outputs = decoder_outputs
        else:
            outputs = other[self.inputs]

        targets = target_variable[self.target]

        for step, step_output in enumerate(outputs):
            step_target = targets[:, step + 1]
            self.eval_step(step_output, step_target)

    def eval_step(self, outputs, target):
        """ Function called by eval batch to evaluate a timestep of the batch.
        When called it updates self.acc_loss with the loss of the current step.

        Args:
            outputs (torch.Tensor): outputs of a batch.
            target (torch.Tensor): expected output of a batch.
        """
        raise NotImplementedError

    def cuda(self):
        self.criterion.cuda()

    def to(self, device):
        self.criterion.to(device)

    def backward(self, retain_graph=False):
        """ Backpropagate the computed loss.
        """
        if type(self.acc_loss) is int:
            raise ValueError("No loss to back propagate.")
        self.acc_loss.backward(retain_graph=retain_graph)

    def scale_loss(self, factor):
        """ Scale loss with a factor
        """
        self.acc_loss*=factor

class NLLLoss(Loss):
    """ Batch averaged negative log-likelihood loss.

    Args:
        ignore_index (int, optional): index of masked token
        size_average (bool, optional): refer to http://pytorch.org/docs/master/nn.html#nllloss
    """

    _NAME = "Avg NLLLoss"
    _SHORTNAME = "nll_loss"
    _INPUTS = "decoder_output"
    _TARGETS = "decoder_output"

    def __init__(self, ignore_index=-1, size_average=True):
        self.ignore_index = ignore_index
        self.size_average = size_average

        super(NLLLoss, self).__init__(
            self._NAME, self._SHORTNAME, self._INPUTS, self._TARGETS,
            nn.NLLLoss(ignore_index=ignore_index, size_average=size_average))

    def get_loss(self):
        if isinstance(self.acc_loss, int):
            return 0
        # total loss for all batches
        loss = self.acc_loss.item()
        if self.size_average:
            # average loss per batch
            loss /= self.norm_term
        return loss

    def eval_step(self, step_outputs, target):
        batch_size = target.size(0)
        outputs = step_outputs.contiguous().view(batch_size, -1)
        self.acc_loss += self.criterion(outputs, target)
        self.norm_term += 1

class Perplexity(NLLLoss):
    """ Language model perplexity loss.

    Perplexity is the token averaged likelihood.  When the averaging options are the
    same, it is the exponential of negative log-likelihood.

    Args:
        ignore_index (int, optional): index to be masked, refer to http://pytorch.org/docs/master/nn.html#nllloss
    """

    _NAME = "Perplexity"
    _SHORTNAME = "ppl"
    _MAX_EXP = 100
    _INPUTS = "decoder_output"

    def __init__(self, ignore_index=-100):
        super(Perplexity, self).__init__(ignore_index=ignore_index, size_average=False)

    def eval_step(self, outputs, target):
        self.acc_loss += self.criterion(outputs, target)
        if self.ignore_index is -100:
            self.norm_term += np.prod(target.size())
        else:
            self.norm_term += target.data.ne(self.ignore_index).sum()

    def get_loss(self):
        nll = super(Perplexity, self).get_loss()
        nll /= self.norm_term.item()
        if nll > Perplexity._MAX_EXP:
            print("WARNING: Loss exceeded maximum value, capping to e^100")
            return math.exp(Perplexity._MAX_EXP)
        return math.exp(nll)
