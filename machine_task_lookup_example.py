import os
import argparse
import logging

import torch
import torchtext
from collections import OrderedDict

from machine.trainer import SupervisedTrainer
from machine.models import EncoderRNN, DecoderRNN, Seq2seq
from machine.loss import NLLLoss
from machine.metrics import SequenceAccuracy
from machine.dataset import SourceField, TargetField

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

# CONSTANTS
IGNORE_INDEX = -1
LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'


def train_lookup_model(default_settings='baseline_2018'):

    if default_settings != 'baseline_2018' and default_settings != 'full_focus_2018':
        raise NameError("Invalid default setting name for lookup table task \n \
        - check .yml file or machine-tasks repo for more info")

    # Add machine-task to path
    import sys
    sys.path.append(os.path.join(os.getcwd(), "machine-tasks"))

    # gets the lookupt task from machine-tasks
    def get_lookup_task_dev():
        from tasks import get_task
        T = get_task("lookup", is_mini=True)
        return T

    # T is lookup task object
    T = get_lookup_task_dev()

    parameters = T.default_params[default_settings]
    train_path = T.train_path
    valid_path = T.valid_path
    test_paths = T.test_paths

    # Prepare logging and data set
    init_logging(parameters)
    src, tgt, train, dev, monitor_data = prepare_dataset(
        parameters, train_path, test_paths, valid_path)

    # Prepare model
    seq2seq, output_vocab = initialize_model(
        parameters, src, tgt, train)

    pad = output_vocab.stoi[tgt.pad_token]

    # Prepare training
    losses = [NLLLoss(ignore_index=pad)]
    for loss in losses:
        loss.to(device)
    loss_weights = [1.]
    metrics = [SequenceAccuracy(ignore_index=pad)]

    trainer = create_trainer(
        parameters['batch_size'], losses, loss_weights, metrics)

    # Train
    seq2seq, _ = trainer.train(seq2seq, train,
                               num_epochs=100, dev_data=dev,
                               monitor_data=monitor_data, optimizer='adam',
                               checkpoint_path='../models')


def init_logging(parameters):
    logging.basicConfig(format=LOG_FORMAT, level=getattr(
        logging, 'INFO'))
    logging.info(parameters)


def prepare_dataset(parameters, train_path, test_paths, valid_path):
    src = SourceField()
    tgt = TargetField(include_eos=False)
    tabular_data_fields = [('src', src), ('tgt', tgt)]

    max_len = parameters['max_len']

    def len_filter(example):
        return len(example.src) <= max_len and len(example.tgt) <= max_len

    # generate training and testing data
    train = torchtext.data.TabularDataset(
        path=train_path, format='tsv',
        fields=tabular_data_fields,
        filter_pred=len_filter
    )

    dev = torchtext.data.TabularDataset(
        path=valid_path, format='tsv',
        fields=tabular_data_fields,
        filter_pred=len_filter
    )

    monitor_data = OrderedDict()
    for dataset in test_paths:
        m = torchtext.data.TabularDataset(
            path=dataset, format='tsv',
            fields=tabular_data_fields,
            filter_pred=len_filter)
        monitor_data[dataset] = m

    return src, tgt, train, dev, monitor_data


def initialize_model(parameters, src, tgt, train):
    # build vocabulary
    src.build_vocab(train, max_size=50000)
    tgt.build_vocab(train, max_size=50000)

    output_vocab = tgt.vocab

    # Initialize model
    hidden_size = parameters['hidden_size']
    encoder = EncoderRNN(len(src.vocab), parameters['max_len'], hidden_size,
                         parameters['embedding_size'],
                         rnn_cell=parameters['rnn_cell'],
                         variable_lengths=True)
    decoder = DecoderRNN(len(tgt.vocab), parameters['max_len'], hidden_size,
                         attention_method=parameters['attention_method'],
                         full_focus=parameters['full_focus'],
                         rnn_cell=parameters['rnn_cell'],
                         eos_id=tgt.eos_id, sos_id=tgt.sos_id)
    seq2seq = Seq2seq(encoder, decoder)
    seq2seq.to(device)

    for param in seq2seq.parameters():
        param.data.uniform_(-0.08, 0.08)

    return seq2seq, output_vocab


def create_trainer(batch_size, losses, loss_weights, metrics):
    return SupervisedTrainer(loss=losses, metrics=metrics,
                             loss_weights=loss_weights, batch_size=batch_size,
                             eval_batch_size=512, checkpoint_every=10,
                             print_every=10, expt_dir='../models')


if __name__ == "__main__":
    train_lookup_model(default_settings='baseline_2018')
    train_lookup_model(default_settings='full_focus_2018')