import torch.nn as nn
import torch.nn.functional as F

import sys
import abc
import types

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(types.StringType('ABC'), (), {})

class BaseModel(ABC, nn.Module):
    """ 
    Abstract base class for models.

    Args:
        encoder_module (baseRNN) :  module that encodes inputs
        decoder_module (baseRNN, optional):   module to decode encoded inputs
        decode_function (callable, optional): function to generate symbols from output hidden states (default: F.log_softmax)

    """

    def __init__(self, encoder_module, decoder_module=None, decode_function=F.log_softmax):
        super(BaseModel, self).__init__()
        self.encoder_module = encoder_module
        self.decoder_module = decoder_module
        self.decode_function = decode_function

    def flatten_parameters(self):
        """
        Flatten parameters of all components in the model.
        """
        raise NotImplementedError("A generic version of this function should be implemented")

    def reset_parameters(self):
        """
        Reset the parameters of all components in the model.
        """
        raise NotImplementedError("A generic version of this function should be implemented")

    def forward(self, inputs, input_lengths=None, targets={},
                teacher_forcing_ratio=0):
        """
        Inputs: inputs, input_lengths, targets, teacher_forcing_ratio
            - **inputs** (list, option): list of sequences, whose length is the batch size and within which
              each sequence is a list of token IDs. This information is passed to the encoder module.
            - **input_lengths** (list of int, optional): A list that contains the lengths of sequences
                in the mini-batch, it must be provided when using variable length RNN (default: `None`)
            - **targets** (list, optional): list of sequences, whose length is the batch size and within which
              each sequence is a list of token IDs. This information is forwarded to the decoder.
            - **teacher_forcing_ratio** (float, optional): The probability that teacher forcing will be used. A random number
              is drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
              teacher forcing would be used (default is 0)

        Outputs: decoder_outputs, decoder_hidden, ret_dict
            - **outputs** (batch): batch-length list of tensors with size (max_length, hidden_size) containing the
              outputs of the decoder.
            - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
              state of the decoder.
            - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
              representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
              predicted token IDs, *KEY_INPUT* : target outputs if provided for decoding, *KEY_ATTN_SCORE* : list of
              sequences, where each list is of attention weights }.
        """
        pass
