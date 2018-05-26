from __future__ import print_function
import math
import torch
import torch.nn as nn
import numpy as np

class Metric(object):
    """ Base class for encapsulation of the metrics functions.

    This class defines interfaces that are commonly used with loss functions
    in training and inferencing.  For information regarding individual loss
    functions, please refer to http://pytorch.org/docs/master/nn.html#loss-functions

    Note:
        Do not use this class directly, use one of the sub classes.

    Attributes:
        name (str): name of the metric used by logging messages.
        target (str): dictionary key to fetch the target from dictionary that stores
                      different variables computed during the forward pass of the model
        metric_total (int or torcn.nn.Tensor): variable that stores accumulated loss.
        norm_term (float): normalization term that can be used to calculate
            the value of the metric of multiple batches.
            sub-classes.
    """

    def __init__(self, name, log_name, input_var):
        self.name = name
        self.log_name = log_name
        self.input = input_var

    def reset(self):
        """ Reset accumulated metric values"""
        raise NotImplementedError("Implement in subclass")

    def get_val(self):
        """ Get the value for the metric given the accumulated loss
        and the normalisation term

        Returns:
            loss (float): value of the metric.
        """
        raise NotImplementedError("Implement in subclass")

    def eval_batch(self, outputs, target):
        """ Compute the metric for the batch given results and target results.

        Args:
            outputs (torch.Tensor): outputs of a batch.
            target (torch.Tensor): expected output of a batch.
        """
        raise NotImplementedError("Implement in subclass")

class WordAccuracy(Metric):
    """
    Batch average of word accuracy.

    Args:
        ignore_index (int, optional): index of masked token
    """

    _NAME = "Word Accuracy"
    _SHORTNAME = "acc"
    _INPUT = "sequence"

    def __init__(self, ignore_index=None):
        self.ignore_index = ignore_index
        self.word_match = 0
        self.word_total = 0

        super(WordAccuracy, self).__init__(self._NAME, self._SHORTNAME, self._INPUT)

    def get_val(self):
        if self.word_total != 0:
            return float(self.word_match)/self.word_total
        else:
            return 0

    def reset(self):
        self.word_match = 0
        self.word_total = 0

    def eval_batch(self, outputs, targets):
        # evaluate batch
        targets = targets['decoder_output']
        batch_size = targets.size(0)

        for step, step_output in enumerate(outputs):
            target = targets[:, step + 1]
            non_padding = target.ne(self.ignore_index)
            correct = outputs[step].view(-1).eq(target).masked_select(non_padding).long().sum().item()
            self.word_match += correct
            self.word_total += non_padding.long().sum().item()

class FinalTargetAccuracy(Metric):
    """
    Batch average of the accuracy on the final target (step before <eos>, if eos is present)

    Args:
        ignore_index (int, optional): index of padding
    """

    _NAME = "Final Target Accuracy"
    _SHORTNAME = "target_acc"
    _INPUT = "sequence"

    def __init__(self, ignore_index, eos_id):    # TODO check if returns error if default is not given
        self.ignore_index = ignore_index
        self.eos = eos_id
        self.word_match = 0
        self.word_total = 0

        super(FinalTargetAccuracy, self).__init__(self._NAME, self._SHORTNAME, self._INPUT)

    def get_val(self):
        if self.target_total != 0:
            return float(self.target_match)/self.target_total
        else:
            return 0

    def reset(self):
        self.target_match = 0
        self.target_total = 0

    def eval_batch(self, outputs, targets):
        # evaluate batch
        targets = targets['decoder_output']
        batch_size = targets.size(0)

        self.target_total += batch_size

        for step, step_output in enumerate(outputs):

            target = targets[:, step + 1]

            # compute mask for current step
            cur_mask = target.ne(self.ignore_index)*target.ne(self.eos) # return 1 only if not equal to pad or <eos>

            # compute whether next step is <eos> or pad
            try:
                target_next = targets[:, step + 2]
                mask_next = target_next.eq(self.ignore_index)+target_next.eq(self.eos)
                mask = mask_next*cur_mask
            except IndexError:
                # IndexError if we are dealing with last step, in case just apply cur mask
                mask = cur_mask

            # compute correct, masking all outputs that are padding or eos, or are not followed by padding or eos
            correct = step_output.view(-1).eq(target).masked_select(mask).long().sum().item()

            self.target_match += correct

class SequenceAccuracy(Metric):
    """
    Batch average of word accuracy.

    Args:
        ignore_index (int, optional): index of masked token
    """

    _NAME = "Sequence Accuracy"
    _SHORTNAME = "seq_acc"
    _INPUT = "seqlist"

    def __init__(self, ignore_index=None):
        self.ignore_index = ignore_index
        self.seq_match = 0
        self.seq_total = 0

        super(SequenceAccuracy, self).__init__(self._NAME, self._SHORTNAME, self._INPUT)

    def get_val(self):
        if self.seq_total != 0:
            return float(self.seq_match)/self.seq_total
        else:
            return 0

    def reset(self):
        self.seq_match = 0
        self.seq_total = 0

    def eval_batch(self, outputs, targets):

        targets = targets['decoder_output']

        batch_size = targets.size(0)

        # compute sequence accuracy over batch
        match_per_seq = torch.zeros(batch_size).type(torch.FloatTensor)
        total_per_seq = torch.zeros(batch_size).type(torch.FloatTensor)

        for step, step_output in enumerate(outputs):
            target = targets[:, step + 1]

            non_padding = target.ne(self.ignore_index)

            correct_per_seq = (outputs[step].view(-1).eq(target)*non_padding)
            # correct_per_seq = (outputs[step].view(-1).eq(target)*non_padding).data
            match_per_seq += correct_per_seq.type(torch.FloatTensor)
            total_per_seq += non_padding.type(torch.FloatTensor)

        self.seq_match += match_per_seq.eq(total_per_seq).long().sum()
        self.seq_total += total_per_seq.shape[0]

class SymbolRewritingAccuracy(Metric):
    """
    Batch average of symbol-rewriting task sequence accuracy.
    This metric is very specific for the symbol rewriting task.
    (see: https://arxiv.org/abs/1805.09657 and https://arxiv.org/pdf/1805.01445)
    For one input, multiple outputs can be correct.

    Args:
        input_vocab (torchtext.vocab): Input dictionary
        output_vocab (torchtext.vocab): Output dictionary
        use_output_eos (bool): Boolean to indicate whether an output EOS should be present
        input_pad_symbol (str): Input PAD symbol
        output_sos_symbol (str): Output SOS symbol
        output_pad_symbol (str): Output PAD symbol
        output_eos_symbol (str): Output EOS symbol
        output_unk_symbol (str): Output UNK symbol
    """

    _NAME = "Symbol Rewriting Accuracy"
    _SHORTNAME = "sym_rwr_acc"
    _INPUT = "seqlist"

    def __init__(self, input_vocab, output_vocab, use_output_eos, input_pad_symbol, output_sos_symbol, output_pad_symbol, output_eos_symbol, output_unk_symbol):
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab

        self.use_output_eos = use_output_eos

        # instead of passing all these arguments, we could also hard-code to use <sos>, <pad>, <unk> and <eos>
        self.input_pad_symbol = input_pad_symbol
        self.output_sos_symbol = output_sos_symbol
        self.output_pad_symbol = output_pad_symbol
        self.output_eos_symbol = output_eos_symbol
        self.output_unk_symbol = output_unk_symbol

        self.seq_correct = 0
        self.seq_total = 0

        super(SymbolRewritingAccuracy, self).__init__(self._NAME, self._SHORTNAME, self._INPUT)

    def get_val(self):
        """
        Get the average accuracy metric of all processed batches
        Returns:
            float: average accuracy
        """
        if self.seq_total != 0:
            return float(self.seq_correct) / self.seq_total
        else:
            return 0

    def reset(self):
        """
        Reset after all batches have been processed
        """
        self.seq_correct = 0
        self.seq_total = 0

    # Original code provided by the authors of The Fine Line between
    # Linguistic Generalization and Failure in Seq2Seq-Attention Models
    # (https://arxiv.org/pdf/1805.01445.pdf)
    def correct(self, grammar, prediction):
        '''
        Return True if the target is a valid output given the source
        Args:
            grammar (list(str)): List of symbols of the grammar
            prediction (list(str)): List of symbols of the prediction
        Returns:
            bool: whether the prediction is coming from the grammar
        '''

        grammar_vocab = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'T', 'U',
                         'V', 'W', 'X', 'Y', 'Z', 'AS', 'BS', 'CS', 'DS', 'ES', 'FS', 'GS', 'HS', 'IS', 'JS', 'KS', 'LS', 'MS', 'NS', 'OS']

        all_correct = False
        # Check if the length is correct
        length_check = True if len(prediction) == 3 * len(grammar) else False
        # Check if everything falls in the same bucket, and there are no repeats
        for idx, inp in enumerate(grammar):
            vocab_idx = grammar_vocab.index(inp) + 1
            span = prediction[idx * 3:idx * 3 + 3]

            span_str = " ".join(span)
            if (not all(int(item.replace("A", "").replace("B", "").replace("C", "").split("_")[0]) == vocab_idx for item in span)
                    or (not ("A" in span_str and "B" in span_str and "C" in span_str))):
                all_correct = False
                break
            else:
                all_correct = True
        return all_correct

    def eval_batch(self, outputs, targets):
        """
        Evaluates one batch of inputs (grammar) and checks whether the predictions are correct in the
        specified grammar.
        Note that we assume that the input grammar's do not contain any EOS-like symbol
        Args:
            outputs (list(torch.tensor)): Contains the predictions of the model. List of length max_output_length, where each element is a tensor of length batch_size
            targets (dict): Dictionary containing the grammars
        """

        # batch_size X N variable containing the indices of the model's input,
        # where N is the longest input
        input_variable = targets['encoder_input']

        batch_size = input_variable.size(0)

        # Convert to batch_size x M variable containing the indices of the model's output, where M
        # is the longest output
        predictions = torch.stack(outputs, dim=1).view(batch_size, -1)

        # Current implementation does not allow batch-wise evaluation
        for i_batch_element in range(batch_size):
            # We start by counting the sequence to the total.
            # Next we go through multiple checks for incorrectness.
            # If all these test fail, we consider the sequence correct.
            self.seq_total += 1

            # Extract the current example and move to cpu
            grammar = input_variable[i_batch_element, :].data.cpu().numpy()
            prediction = predictions[i_batch_element, :].data.cpu().numpy()

            # Convert indices to strings
            # Remove all padding from the grammar.
            grammar = [self.input_vocab.itos[token] for token in grammar if token !=
                       self.input_vocab.itos[token] != self.input_pad_symbol]
            prediction = [self.output_vocab.itos[token] for token in prediction]

            # Each input symbol has to produce exactly three outputs
            required_output_length = 3 * len(grammar)

            # The first prediction after the actual output should be EOS
            if self.use_output_eos and prediction[required_output_length] != self.output_eos_symbol:
                continue

            # Remove EOS (and possible padding)
            prediction_correct_length = prediction[:required_output_length]

            # If the EOS symbol is present in the prediction, this means that the prediction was too
            # short.
            # Since SOS, PAD and UNK are also part of the output dictionary, these can technically
            # also be predicted by the model, especially at the beginning of training due to random
            # weight initialization. Since these render the output incorrect and cause an error in
            # correct(), we check for their presence here.
            if  self.output_eos_symbol in prediction_correct_length or \
                    self.output_sos_symbol in prediction_correct_length or \
                    self.output_pad_symbol in prediction_correct_length or \
                    self.output_unk_symbol in prediction_correct_length:
                continue

            # Check whether the prediction actually comes from the grammar
            if self.correct(grammar, prediction_correct_length):
                self.seq_correct += 1