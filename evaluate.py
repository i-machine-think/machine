import os
import argparse
import logging

import torch
import torchtext

import seq2seq
from seq2seq.loss import Perplexity, AttentionLoss, NLLLoss
from seq2seq.metrics import WordAccuracy, SequenceAccuracy, FinalTargetAccuracy
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Evaluator
from seq2seq.trainer import SupervisedTrainer, LookupTableAttention, LookupTablePonderer, AttentionTrainer
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.trainer import SupervisedTrainer

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint_path', help='Give the checkpoint path from which to load the model')
parser.add_argument('--test_data', help='Path to test data')
parser.add_argument('--cuda_device', default=0, type=int, help='set cuda device to use')
parser.add_argument('--max_len', type=int, help='Maximum sequence length', default=50)
parser.add_argument('--batch_size', type=int, help='Batch size', default=32)

parser.add_argument('--pondering', action='store_true')

parser.add_argument('--attention', choices=['pre-rnn', 'post-rnn'], default=False)
parser.add_argument('--attention_method', choices=['dot', 'mlp'], default=None)
parser.add_argument('--use_attention_loss', action='store_true')
parser.add_argument('--scale_attention_loss', type=float, default=1.)

parser.add_argument('--use_input_eos', action='store_true', help='EOS symbol in input sequences is not used by default. Use this flag to enable.')
parser.add_argument('--ignore_output_eos', action='store_true', help='Ignore end of sequence token during training and evaluation')

opt = parser.parse_args()

IGNORE_INDEX=-1
output_eos_used= not opt.ignore_output_eos

if torch.cuda.is_available():
        print("Cuda device set to %i" % opt.cuda_device)
        torch.cuda.set_device(opt.cuda_device)

#################################################################################
# load model

logging.info("loading checkpoint from {}".format(os.path.join(opt.checkpoint_path)))
checkpoint = Checkpoint.load(opt.checkpoint_path)
seq2seq = checkpoint.model
input_vocab = checkpoint.input_vocab
output_vocab = checkpoint.output_vocab

############################################################################
# Prepare dataset and loss
src = SourceField(opt.use_input_eos)
tgt = TargetField(output_eos_used)
src.vocab = input_vocab
tgt.vocab = output_vocab
tgt.eos_id = tgt.vocab.stoi[tgt.SYM_EOS]
tgt.sos_id = tgt.vocab.stoi[tgt.SYM_SOS]
src.eos_id = src.vocab.stoi[src.SYM_EOS]
max_len = opt.max_len

def len_filter(example):
    return len(example.src) <= max_len and len(example.tgt) <= max_len

# generate test set
test = torchtext.data.TabularDataset(
    path=opt.test_data, format='tsv',
    fields=[('src', src), ('tgt', tgt)],
    filter_pred=len_filter
)

# Prepare loss and metrics
pad = output_vocab.stoi[tgt.pad_token]
losses = [NLLLoss(ignore_index=pad)]
loss_weights = [1.]

if opt.use_attention_loss:
    losses.append(AttentionLoss(ignore_index=IGNORE_INDEX))
    loss_weights.append(opt.scale_attention_loss)

metrics = [WordAccuracy(ignore_index=pad), SequenceAccuracy(ignore_index=pad), FinalTargetAccuracy(ignore_index=pad, eos_id=tgt.eos_id)]
if torch.cuda.is_available():
    for loss in losses:
        loss.cuda()

# Initialize ponderer and attention guidance
ponderer = None
data_func = SupervisedTrainer.get_batch_data
get_batch_kwargs = {}
if opt.pondering:
    ponderer = LookupTablePonderer(input_eos_used=opt.use_input_eos, output_eos_used=output_eos_used)
attention_function = None
if opt.use_attention_loss:
    attention_function = LookupTableAttention(pad_value=IGNORE_INDEX, input_eos_used=opt.use_input_eos, output_eos_used=output_eos_used)
    data_func = AttentionTrainer.get_batch_data
    get_batch_kwargs['attention_function'] = attention_function


#################################################################################
# Evaluate model on test set

evaluator = Evaluator(batch_size=opt.batch_size, loss=losses, metrics=metrics)
losses, metrics = evaluator.evaluate(model=seq2seq, data=test, get_batch_data=data_func, ponderer=ponderer, **get_batch_kwargs)

total_loss, log_msg, _ = SupervisedTrainer.get_losses(losses, metrics, 0)

print(log_msg)
