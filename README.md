# machine &middot; [![Build Status](https://travis-ci.org/i-machine-think/machine.svg?branch=master)](https://travis-ci.org/i-machine-think/machine)

[Documentation](https://i-machine-think.github.io/machine/build/html/index.html)

# Introduction

This is a pytorch implementation of a sequence to sequence learning toolkit for the i-machine-think project. This repository is a fork from the pytorch-seq2seq library developed by IBM, but has substantially diverged from it after heavy development. For the original implementation, visit [https://github.com/IBM/pytorch-seq2seq](https://github.com/IBM/pytorch-seq2seq).

# Requirements

This library runs with PyTorch 0.3.0. We refer to the [PyTorch website](http://pytorch.org/) to install the right version for your environment.
To install additional requirements (including numpy and torchtext), run:

`pip install requirements`

# Quickstart

There are 3 commandline tools available

* `train_model.py`
* `evaluate.py`
* `infer.py`

### Training

The script `train_model.py` can be used to train a new model, resume the training of an existing model from a checkpoint, or retrain an existing model from a checkpoint. E.g. to train a model from scratch:

     # Train a simple model with hidden layer size 128 and embedding size 128
    `python train_model.py --train $train_path --dev $dev_path --output_dir $expt_dir  --embedding_size 128 --hidden_size 256 --rnn_cell gru --epoch 20 

Several options are available from the command line, including changing the optimizer, batch size, using attention/bidirectionality and using teacher forcing. 
For a complete overview, use the *help* function of the script.

### Evaluation and inference

The scripts `infer.py` and `evaluate.py` can be used to run an existing model (loaded from a checkpoint) in inference mode, and evaluate a model on a test set, respectively. E.g: 

      # Use the model stored in $checkpoint_path in inference mode
    ` python infer.py --checkpoint_path $checkpoint_path
    
      # Evaluate a trained model stored in $checkpoint_path
    ` python evaluate.py --checkpoint_path $checkpoint_path --test_data $test_data

### Example script

The script `example.sh` illustrates the usage of all three tools: it uses the toy data from the test directory (containing a 'reverse' dataset in which the translation of any sequence of numbers is its inverse), trains a model on this data using `train_model.py`, evaluates this model using `evaluate.py` and then runs `infer.py` to generate outputs.

Once training is complete, you will be prompted to enter a new sequence to translate and the model will print out its prediction (use ctrl-C to terminate).  Try the example below!

    Input:  1 3 5 7 9
	Expected output: 9 7 5 3 1 EOS


### Checkpoints

During training, the top *k* models are stored in a folder which is named using the accuracy and loss of the model on the development set.
Currently, *k* is set to 5.
The folder contains the model, the source and target vocabulary and the trainer states.

# Pull requests

We welcome pull requests for the library.
Please run both the unittests and integration test before committing:

`python -m unittest discover`
`sh integration_test.sh`


