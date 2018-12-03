from .baseModel import BaseModel
from .EncoderRNN import EncoderRNN

import torch.nn as nn


class LanguageModel(BaseModel):
    """
    Implements a language model
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
        initrange = 0.1
        self.encoder_module.embedding.weight.data.uniform_(
            -initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        output, hidden = self.encoder_module(input, hidden=hidden)
        output = self.decoder_dropout(output)
        decoded = self.decoder(output.contiguous().view(-1, output.size(2)))

        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.n_layers, batch_size, self.hidden_size),
                weight.new_zeros(self.n_layers, batch_size, self.hidden_size))
