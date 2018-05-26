import os
import argparse
import logging

import torch

import seq2seq
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint_path', help='Give the checkpoint path from which to load the model')
parser.add_argument('--cuda_device', default=0, type=int, help='Set cuda device to use')
parser.add_argument('--debug', action='store_true', help=argparse.SUPPRESS)
parser.add_argument('--log-level', default='info', help='Logging level.')

opt = parser.parse_args()

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)


if torch.cuda.is_available():
        logging.info("Cuda device set to %i" % opt.cuda_device)
        torch.cuda.set_device(opt.cuda_device)

#################################################################################
# load model

logging.info("loading checkpoint from {}".format(os.path.join(opt.checkpoint_path)))
checkpoint = Checkpoint.load(opt.checkpoint_path)
seq2seq = checkpoint.model
input_vocab = checkpoint.input_vocab
output_vocab = checkpoint.output_vocab

#################################################################################
# Generate predictor

predictor = Predictor(seq2seq, input_vocab, output_vocab)

if opt.debug:
    exit()

while True:
        seq_str = raw_input("Type in a source sequence:")
        seq = seq_str.strip().split()
        print(predictor.predict(seq))
