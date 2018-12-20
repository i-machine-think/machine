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
from machine.tasks import get_task
from machine.dataset.get_standard_iter import get_standard_iter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CONSTANTS
IGNORE_INDEX = -1
LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'


# Example for how to run file:
#   python machine_task_lookup_example.py --default_settings "baseline_2018"
#     - uses the baseline parameters stored in machine-task found
#       by Hupkes et al. 2018
#   python machine_task_lookup_example.py --default_settings "Hupkes_2018"
#     - Final settings used in Learning compositionally
#       through attentive guidance(Hupkes et al. 2018)

#       This is just an example of how to use machine.task to train with
#       machine, for more machine specific utility please refer to
#       train_model.py. A lot of machine parameters have been fixed
#       for the sake of keeping this example brief and simple.

def train_lookup_model():
    parser = init_argparser()
    opt = parser.parse_args()
    default_settings = opt.default_settings

    # Add machine-task to path

    # gets the lookupt task from tasks
    T = get_task("lookup", is_mini=True)

    print("Got Task")

    parameters = T.default_params[default_settings]
    train_path = T.train_path
    valid_path = T.valid_path
    test_paths = T.test_paths

    # # Prepare logging and data set
    init_logging(parameters)
    src, tgt, train, dev, monitor_data = prepare_iters(
        parameters, train_path, test_paths, valid_path, parameters['batch_size'])

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

    trainer = SupervisedTrainer(expt_dir='../models')

    # Train
    print("Training")
    seq2seq, _ = trainer.train(seq2seq, train,
                               num_epochs=20, dev_data=dev,
                               monitor_data=monitor_data, optimizer='adam',
                               checkpoint_path='../models',
                               losses=losses, metrics=metrics,
                               loss_weights=loss_weights,
                               checkpoint_every=10,
                               print_every=10)


def init_argparser():
    """
    Args: default_settings (str, optional):

    """
    parser = argparse.ArgumentParser()

    # parser arguments
    #  - baseline_2018: uses the baseline parameters stored in machine-task
    #                  found by Hupkes et al. 2018
    #  - Hupkes_2018: Final settings used in Learning compositionally
    #                 through attentive guidance(Hupkes et al. 2018)
    parser.add_argument('--default_settings', type=str, help='Choose default settings',
                        choices=['baseline_2018', 'Hupkes_2018'], default='baseline_2018')
    return parser


def init_logging(parameters):
    logging.basicConfig(format=LOG_FORMAT, level=getattr(
        logging, 'INFO'))
    logging.info(parameters)


def prepare_iters(parameters, train_path, test_paths, valid_path, batch_size, eval_batch_size=512):
    src = SourceField()
    tgt = TargetField(include_eos=False)
    tabular_data_fields = [('src', src), ('tgt', tgt)]

    max_len = parameters['max_len']

    def len_filter(example):
        return len(example.src) <= max_len and len(example.tgt) <= max_len

    # generate training and testing data
    train = get_standard_iter(torchtext.data.TabularDataset(
        path=train_path, format='tsv',
        fields=tabular_data_fields,
        filter_pred=len_filter
    ), batch_size=batch_size)

    dev = get_standard_iter(torchtext.data.TabularDataset(
        path=valid_path, format='tsv',
        fields=tabular_data_fields,
        filter_pred=len_filter
    ), batch_size=eval_batch_size)

    monitor_data = OrderedDict()
    for dataset in test_paths:
        m = get_standard_iter(torchtext.data.TabularDataset(
            path=dataset, format='tsv',
            fields=tabular_data_fields,
            filter_pred=len_filter), batch_size=eval_batch_size)
        monitor_data[dataset] = m

    return src, tgt, train, dev, monitor_data


def initialize_model(parameters, src, tgt, train):
    # build vocabulary
    src.build_vocab(train.dataset, max_size=50000)
    tgt.build_vocab(train.dataset, max_size=50000)

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


if __name__ == "__main__":
    train_lookup_model()
