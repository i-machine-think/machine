# machine &middot; [![Build Status](https://travis-ci.org/i-machine-think/machine.svg?branch=master)](https://travis-ci.org/i-machine-think/machine)

[Documentation](https://i-machine-think.github.io/machine/build/html/index.html)

# Introduction

This is a pytorch implementation of a sequence to sequence learning toolkit for the i-machine-think project. This repository is a fork from the pytorch-seq2seq library developed by IBM, but has substantially diverged from it after heavy development. For the original implementation, visit [https://github.com/IBM/pytorch-seq2seq](https://github.com/IBM/pytorch-seq2seq).

# Requirements

This library runs with PyTorch 0.3.0. We refer to the [PyTorch website](http://pytorch.org/) to install the right version for your environment.
To install additional requirements (including numpy and torchtext), run:

`pip install -r requirements.txt`

# Quickstart

There are 3 commandline tools available

* `train_model.py`
* `evaluate.py`
* `infer.py`

## Training

The script `train_model.py` can be used to train a new model, resume the training of an existing model from a checkpoint, or retrain an existing model from a checkpoint. E.g. to train a model from scratch:

     # Train a simple model with hidden layer size 128 and embedding size 128
    `python train_model.py --train $train_path --dev $dev_path --output_dir $expt_dir  --embedding_size 128 --hidden_size 256 --rnn_cell gru --epoch 20 

Several options are available from the command line, including changing the optimizer, batch size, using attention/bidirectionality and using teacher forcing. 
For a complete overview, use the *help* function of the script.

## Evaluation and inference

The scripts `infer.py` and `evaluate.py` can be used to run an existing model (loaded from a checkpoint) in inference mode, and evaluate a model on a test set, respectively. E.g: 

      # Use the model stored in $checkpoint_path in inference mode
    ` python infer.py --checkpoint_path $checkpoint_path
    
      # Evaluate a trained model stored in $checkpoint_path
    ` python evaluate.py --checkpoint_path $checkpoint_path --test_data $test_data

## Example script

The script `example.sh` illustrates the usage of all three tools: it uses the toy data from the test directory (containing a 'reverse' dataset in which the translation of any sequence of numbers is its inverse), trains a model on this data using `train_model.py`, evaluates this model using `evaluate.py` and then runs `infer.py` to generate outputs.

Once training is complete, you will be prompted to enter a new sequence to translate and the model will print out its prediction (use ctrl-C to terminate).  Try the example below!

    Input:  1 3 5 7 9
	Expected output: 9 7 5 3 1 EOS


## Checkpoints

During training, the top *k* models are stored in a folder which is named using the accuracy and loss of the model on the development set.
Currently, *k* is set to 5.
The folder contains the model, the source and target vocabulary and the trainer states.

# Contributing

If you have any questions, bug reports and feature requests, please [open an issue](https://github.com/i-machine-think/machine/issues/new) on Github. We welcome any kind of feedback or contribution. Do you want to contribute changes to the codebase? Make sure to read this first.

## Bug reporting

Have you found a bug in the library? Follow these steps to report it:
1. Make sure you have the most recent version of the library. In case you forked the library, consider adding the [original library](https://github.com/i-machine-think/machine/tree/master) as a [remote](https://help.github.com/categories/managing-remotes/) to your local version to keep it up to date.
2. Check if an issue report for your bug already exists.
3. File an issue report, make sure you include the useful information necessary to reproduce the issue, e.g.:
    * What OS are you using?
    * Are you running on GPU? If so, what is your version of Cuda?
4. Provide a script to reproduce the issue. This script should be runnable as-is and should not require external data to download (use randomly generated data if you need to run a model on some test data). We recommend that you use Github Gists to post your code.
5. If possible, try to fix the bug yourself!

## Requesting a feature

You can also use Github issues to request features you would like to have added, or changes in the API. When you do so:
1. Provide a clear and detailed explanation of the feature you want and why it is important.
2. Provide code snippets demonstrating the API you have in mind and illustrating use cases of your feature (writing real code is not required).
3. You may choose to attempt a Pull Request (see below) to include the feature yourself.

## Pull Requests

Before doing a pull request, create an issue describing the feature you want to implement/the bug you have found. When writing the code, please adhere to the following guidelines:
* Include proper docstrings for any new function or class you introduce. They should be formatted in MarkDown and there should be sections for `Args` and `Inputs` and `Outputs`. If applicable, provide also an example. We refer to the docstrings already existing in the codebase for more examples.
* Write tests. Your code should have full unit test coverage. If you want to see your PR merged promptly, this is crucial. For testing, we use unittest in combination with the library [mock](https://docs.python.org/3/library/unittest.mock.html). Please note that all unittests as well as an integration test will be ran automatically when you put your PR, to assure your code does not break any existing functionality. Please run both the unittests and integration test before committing:

        python -m unittest discover
        sh integration_test.sh

* When committing, use appropriate, descriptive commit messages. 


