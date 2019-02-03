from .baseModel import BaseModel
from .EncoderRNN import EncoderRNN

import torch.nn as nn


class LanguageModel(BaseModel):
    """
    Implements a language model

    Args:
        encoder_module (EncoderRNN): Encoder to use
        tie_weights (bool, optional): Whether to tie embedding weights to decoder weights
        dropout_p_decoder (float, optional): dropout prob of decoder

    Inputs: inputs, hidden
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is a list of token IDs.
        - **hidden** : Tuple of (h_0, c_0), each of shape (num_layers * num_directions, batch, hidden_size)
              where h_0 is tensor containing the initial hidden state, and c_0 is a tensor
              containing the initial cell state for for each element in the batch. 

    Outputs: output, hidden
        - **output** (batch, seq_len, hidden_size): tensor containing the decoded features of the input sequence
        - **hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the features in the hidden state `h`
    """

    def __init__(self, encoder_module, tie_weights=False, dropout_p_decoder=0.5):

        super(LanguageModel, self).__init__(encoder_module=encoder_module)

        self.decoder_dropout = nn.Dropout(dropout_p_decoder)
        self.decoder = nn.Linear(
            self.encoder_module.hidden_size, self.encoder_module.vocab_size)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if self.encoder_module.embedding_size != self.encoder_module.hidden_size:
                raise ValueError(
                    'When using the tied flag, encoder embedding_size must be equal to hidden_size')
            self.decoder.weight = self.encoder_module.embedding.weight

        self.init_weights()

        self.hidden_size = self.encoder_module.hidden_size
        self.n_layers = self.encoder_module.n_layers

    def flatten_parameters(self):
        """
        Flatten parameters of all reccurrent components in the model.
        """
        self.encoder_module.rnn.flatten_parameters()

    def init_weights(self):
        """
        Standard weight initialization
        """
        initrange = 0.1
        self.encoder_module.embedding.weight.data.uniform_(
            -initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        output, hidden = self.encoder_module(input, hidden=hidden)
        output = self.decoder_dropout(output)
        decoded = self.decoder(output.contiguous().view(-1, output.size(2)))

        return decoded.view(output.size(0), output.size(1),
                            decoded.size(1)), hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.n_layers, batch_size, self.hidden_size),
                weight.new_zeros(self.n_layers, batch_size, self.hidden_size))
