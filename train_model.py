import os
import argparse
import logging

import torch
from torch.optim.lr_scheduler import StepLR
import torchtext

import random

import seq2seq
from seq2seq.trainer import SupervisedTrainer, LookupTableAttention, AttentionTrainer, LookupTablePonderer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import Perplexity, AttentionLoss, NLLLoss
from seq2seq.metrics import WordAccuracy, SequenceAccuracy, FinalTargetAccuracy
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Predictor, Evaluator
from seq2seq.util.checkpoint import Checkpoint

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

parser = argparse.ArgumentParser()
parser.add_argument('--train', help='Training data')
parser.add_argument('--dev', help='Development data')
parser.add_argument('--output_dir', default='../models', help='Path to model directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--epochs', type=int, help='Number of epochs', default=6)
parser.add_argument('--optim', type=str, help='Choose optimizer', choices=['adam', 'adadelta', 'adagrad', 'adamax', 'rmsprop', 'sgd'])
parser.add_argument('--max_len', type=int, help='Maximum sequence length', default=50)
parser.add_argument('--rnn_cell', help="Chose type of rnn cell", default='lstm')
parser.add_argument('--bidirectional', action='store_true', help="Flag for bidirectional encoder")
parser.add_argument('--embedding_size', type=int, help='Embedding size', default=128)
parser.add_argument('--hidden_size', type=int, help='Hidden layer size', default=128)
parser.add_argument('--n_layers', type=int, help='Number of RNN layers in both encoder and decoder', default=1)
parser.add_argument('--src_vocab', type=int, help='source vocabulary size', default=50000)
parser.add_argument('--tgt_vocab', type=int, help='target vocabulary size', default=50000)
parser.add_argument('--dropout_p_encoder', type=float, help='Dropout probability for the encoder', default=0.2)
parser.add_argument('--dropout_p_decoder', type=float, help='Dropout probability for the decoder', default=0.2)
parser.add_argument('--teacher_forcing_ratio', type=float, help='Teacher forcing ratio', default=0.2)
parser.add_argument('--pondering', action='store_true')
parser.add_argument('--attention', choices=['pre-rnn', 'post-rnn'], default=False)
parser.add_argument('--attention_method', choices=['dot', 'mlp'], default=None)
parser.add_argument('--use_attention_loss', action='store_true')
parser.add_argument('--scale_attention_loss', type=float, default=1.)
parser.add_argument('--batch_size', type=int, help='Batch size', default=32)
parser.add_argument('--lr', type=float, help='Learning rate, recommended settings.\nrecommended settings: adam=0.001 adadelta=1.0 adamax=0.002 rmsprop=0.01 sgd=0.1', default=0.001)

parser.add_argument('--load_checkpoint', help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--save_every', type=int, help='Every how many batches the model should be saved', default=100)
parser.add_argument('--print_every', type=int, help='Every how many batches to print results', default=100)
parser.add_argument('--resume', action='store_true', help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', default='info', help='Logging level.')
parser.add_argument('--cuda_device', default=0, type=int, help='set cuda device to use')
parser.add_argument('--ignore_eos', action='store_true', help='Ignore end of sequence value during trainng and evaluation')

opt = parser.parse_args()

IGNORE_INDEX=-1

if opt.resume and not opt.load_checkpoint:
    parser.error('load_checkpoint argument is required to resume training from checkpoint')

if opt.use_attention_loss and not opt.attention:
    parser.error('Specify attention type to use attention loss')

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

if torch.cuda.is_available():
        print("Cuda device set to %i" % opt.cuda_device)
        torch.cuda.set_device(opt.cuda_device)

if opt.attention:
    if not opt.attention_method:
        opt.attention_method = 'dot'

############################################################################
# Prepare dataset
src = SourceField()
tgt = TargetField()
max_len = opt.max_len

def len_filter(example):
    return len(example.src) <= max_len and len(example.tgt) <= max_len

# generate training and testing data
train = torchtext.data.TabularDataset(
    path=opt.train, format='tsv',
    fields=[('src', src), ('tgt', tgt)],
    filter_pred=len_filter
)

if opt.dev:
    dev = torchtext.data.TabularDataset(
        path=opt.dev, format='tsv',
        fields=[('src', src), ('tgt', tgt)],
        filter_pred=len_filter
    )
else:
    dev = None

#################################################################################
# prepare model

if opt.load_checkpoint is not None:
    logging.info("loading checkpoint from {}".format(os.path.join(opt.output_dir, opt.load_checkpoint)))
    checkpoint_path = os.path.join(opt.output_dir, opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab
    src.vocab = input_vocab
    tgt.vocab = output_vocab
    if opt.ignore_eos:
        assert output_vocab.stoi[tgt.pad_token] == output_vocab.stoi[tgt.eos_token], "train_model.py was called with flag ignore_eos, but eos token is not equal to padding in inputted model"

else:
    # build vocabulary
    src.build_vocab(train, max_size=opt.src_vocab)
    tgt.build_vocab(train, max_size=opt.tgt_vocab)
    # if <eos> should be ignored, set eos equal to pad token
    if opt.ignore_eos:
        tgt.vocab.stoi[tgt.eos_token] = tgt.vocab.stoi[tgt.pad_token]
    input_vocab = src.vocab
    output_vocab = tgt.vocab

    # Initialize model
    hidden_size = opt.hidden_size
    decoder_hidden_size = hidden_size*2 if opt.bidirectional else hidden_size
    encoder = EncoderRNN(len(src.vocab), max_len, hidden_size,
                         opt.embedding_size,
                         dropout_p=opt.dropout_p_encoder,
                         n_layers=opt.n_layers,
                         bidirectional=opt.bidirectional,
                         rnn_cell=opt.rnn_cell,
                         variable_lengths=True)
    decoder = DecoderRNN(len(tgt.vocab), max_len, decoder_hidden_size,
                         dropout_p=opt.dropout_p_decoder,
                         n_layers=opt.n_layers,
                         use_attention=opt.attention,
                         attention_method=opt.attention_method,
                         bidirectional=opt.bidirectional,
                         rnn_cell=opt.rnn_cell,
                         eos_id=tgt.eos_id, sos_id=tgt.sos_id)
    seq2seq = Seq2seq(encoder, decoder)
    if torch.cuda.is_available():
        seq2seq.cuda()

    for param in seq2seq.parameters():
        param.data.uniform_(-0.08, 0.08)

input_vocabulary = input_vocab.itos
output_vocabulary = output_vocab.itos

# random.seed(3)
# 
# print "Input vocabulary:"
# for i, word in enumerate(input_vocabulary):
#     print i, word
# 
# print "Output vocabulary:"
# for i, word in enumerate(output_vocabulary):
#     print i, word
# 
# raw_input()

##############################################################################
# train model

# Prepare loss and metrics
pad = output_vocab.stoi[tgt.pad_token]
loss = [NLLLoss(ignore_index=pad)]
loss_weights = [1.]

if opt.use_attention_loss:
    loss.append(AttentionLoss(ignore_index=IGNORE_INDEX))
    loss_weights.append(opt.scale_attention_loss)

metrics = [WordAccuracy(ignore_index=pad), SequenceAccuracy(ignore_index=pad), FinalTargetAccuracy(ignore_index=pad, eos_token=tgt.eos_id)]
if torch.cuda.is_available():
    for loss_func in loss:
        loss_func.cuda()

checkpoint_path = os.path.join(opt.output_dir, opt.load_checkpoint) if opt.resume else None

ponderer = None
if opt.pondering:
    ponderer = LookupTablePonderer()
if opt.use_attention_loss:
    attention_function = LookupTableAttention(pad_value=IGNORE_INDEX)

# create trainer
if not opt.use_attention_loss:
    t = SupervisedTrainer(loss=loss, metrics=metrics, 
                          loss_weights=loss_weights,
                          batch_size=opt.batch_size,
                          checkpoint_every=opt.save_every,
                          print_every=opt.print_every, expt_dir=opt.output_dir)

    seq2seq = t.train(seq2seq, train, 
                      num_epochs=opt.epochs, dev_data=dev,
                      ponderer=ponderer,
                      optimizer=opt.optim,
                      teacher_forcing_ratio=opt.teacher_forcing_ratio,
                      learning_rate=opt.lr,
                      resume=opt.resume,
                      checkpoint_path=checkpoint_path)
else:
    t = AttentionTrainer(loss=loss, metrics=metrics, 
                          loss_weights=loss_weights,
                          batch_size=opt.batch_size,
                          checkpoint_every=opt.save_every,
                          print_every=opt.print_every, expt_dir=opt.output_dir)

    seq2seq = t.train(seq2seq, train, 
                      num_epochs=opt.epochs, dev_data=dev,
                      attention_function=attention_function,
                      ponderer=ponderer,
                      optimizer=opt.optim,
                      teacher_forcing_ratio=opt.teacher_forcing_ratio,
                      learning_rate=opt.lr,
                      resume=opt.resume,
                      checkpoint_path=checkpoint_path)

# evaluator = Evaluator(loss=loss, batch_size=opt.batch_size)
# dev_loss, accuracy = evaluator.evaluate(seq2seq, dev)
