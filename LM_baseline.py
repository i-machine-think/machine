from torchtext.datasets import WikiText2
from tqdm import tqdm
import os
import argparse
import logging

import torch
import torchtext
import torch.nn as nn
import torch.optim as optim
import time
from collections import OrderedDict

from machine.trainer import SupervisedTrainer
from machine.models import EncoderRNN, DecoderRNN, LanguageModel
from machine.loss import Perplexity

import math


class Baseline_LSTM_LM(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(Baseline_LSTM_LM, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError(
                    'When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(
            output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


batch_size = 32
lr = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_iter, valid_iter, test_iter = WikiText2.iters(
    batch_size=batch_size, device=device)
vocab_size = len(train_iter.dataset.fields['text'].vocab)
model = Baseline_LSTM_LM(vocab_size, 64, 64, 1)
optimizer = optim.Adam(model.parameters(), lr=lr)

criterion = nn.CrossEntropyLoss()


model.train()
total_loss = 0.0
log_interval = 10
start_time = time.time()
hidden = model.init_hidden(batch_size)
for i, batch in tqdm(enumerate(train_iter)):

    data, targets = batch.text, batch.target

    # Starting each batch, we detach the hidden state from how it was previously produced.
    # If we didn't, the model would try backpropagating all the way to start of the dataset.
    hidden = repackage_hidden(hidden)
    model.zero_grad()
    output, hidden = model(data, hidden)
    loss = criterion(output.view(-1, vocab_size), targets.view(-1))
    loss.backward()

    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
    for p in model.parameters():
        p.data.add_(-lr, p.grad.data)

    total_loss += loss.item()

    if i % log_interval == 0 and i > 0:
        cur_loss = total_loss / log_interval
        elapsed = time.time() - start_time
        print('loss {:5.2f} | ppl {:8.2f}'.format(
            cur_loss, math.exp(cur_loss)))
        # print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
        #       'loss {:5.2f} | ppl {:8.2f}'.format(
        #           epoch, batch, len(train_data) // 35, lr,
        #           elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
        total_loss = 0
