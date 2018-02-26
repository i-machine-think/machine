from __future__ import division
import logging
import os
import random
import time
import shutil

import torch
import torchtext
from torch import optim

import seq2seq
from seq2seq.evaluator import Evaluator
from seq2seq.loss import NLLLoss, AttentionLoss
from .supervised_trainer import SupervisedTrainer
from seq2seq.optim import Optimizer
from seq2seq.util.checkpoint import Checkpoint

class AttentionTrainer(SupervisedTrainer):
    """ The AttentionTrainer class is helps in setting up a training framework to
    train a model with attentive guidance.

    Args:
        expt_dir (optional, str): experiment Directory to store details of the experiment,
            by default it makes a folder in the current directory to store the details (default: `experiment`).
        loss (seq2seq.loss.loss.Loss, optional): loss for training, (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for experiment, (default: 64)
        checkpoint_every (int, optional): number of epochs to checkpoint after, (default: 100)
    """
    def __init__(self, expt_dir='experiment', loss=[NLLLoss(), AttentionLoss()],
                 loss_weights=None, metrics=[], batch_size=64, random_seed=None,
                 checkpoint_every=100, print_every=100):
        self._trainer = "Attention Trainer"
        super(AttentionTrainer, self).__init__(expt_dir=expt_dir, loss=loss, loss_weights=loss_weights, metrics=metrics, batch_size=batch_size, random_seed=random_seed, checkpoint_every=checkpoint_every, print_every=print_every)
        
    def train(self, model, data, num_epochs=5,
              resume=False, dev_data=None,
              attention_function=None, ponderer=None,
              optimizer=None, teacher_forcing_ratio=0,
              learning_rate=0.001, checkpoint_path=None, top_k=5):

        self.attention_function = attention_function
        if self.attention_function is None:
            raise ValueError("attention_function cannot be None in attention trainer, provide function to generate attention targets")

        super(AttentionTrainer, self).train(
                model=model, data=data, 
                ponderer=ponderer,
                num_epochs=num_epochs, resume=resume,
                dev_data=dev_data, optimizer=optimizer,
                teacher_forcing_ratio=teacher_forcing_ratio,
                learning_rate=learning_rate, checkpoint_path=checkpoint_path,
                top_k=top_k)

    def get_batch_data(self, batch):
        input_variables, input_lengths, target_variables = super(AttentionTrainer, self).get_batch_data(batch)
        target_variables = self.attention_function.add_attention_targets(input_variables, input_lengths, target_variables)

        return input_variables, input_lengths, target_variables
