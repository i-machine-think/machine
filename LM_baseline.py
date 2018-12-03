from torchtext.datasets import WikiText2, PennTreebank
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

# from machine.trainer import SupervisedTrainer
from machine.models import EncoderRNN, LanguageModel

import math


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


batch_size = 64
lr = 0.001
lr_decay = 0.1
NUM_EPOCHS = 40
log_interval = 100

torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# train_iter, valid_iter, test_iter = WikiText2.iters(
# batch_size=batch_size, device=device)


train_iter, valid_iter, test_iter = PennTreebank.iters(
    batch_size=batch_size, device=device)

vocab_size = len(train_iter.dataset.fields['text'].vocab)

encoder = EncoderRNN(vocab_size, 35, 650, 650,
                     rnn_cell='lstm', input_dropout_p=0.5, dropout_p=0.5)
model = LanguageModel(encoder, tie_weights=True, dropout_p_decoder=0.5)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr,
                       betas=(0.0, 0.999), eps=1e-8, weight_decay=12e-7)

criterion = nn.CrossEntropyLoss()


def evaluate(data_source):
    # data_source = valid_iter or test_iter
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    hidden = model.init_hidden(batch_size)
    example_count = 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(data_source), total=len(data_source)):
            data, targets = batch.text.t(), batch.target
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, vocab_size)
            total_loss += len(data) * criterion(output_flat,
                                                targets.view(-1)).item()
            example_count += len(data)
            hidden = repackage_hidden(hidden)

    return total_loss / example_count


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    hidden = model.init_hidden(batch_size)
    for i, batch in tqdm(enumerate(train_iter), total=len(train_iter)):
        # transpose text to make batch first
        data, targets = batch.text.t(), batch.target

        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, vocab_size), targets.view(-1))
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()

        # We detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)

        total_loss += loss.item()

        if i % log_interval == 0 and i > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                      epoch, i, len(train_iter) // 35, lr,
                      elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


# Loop over epochs.
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, NUM_EPOCHS+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(valid_iter)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open('model.pt', 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            print("No improvement in val loss - decreasing lr")
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open('model.pt', 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_iter)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
