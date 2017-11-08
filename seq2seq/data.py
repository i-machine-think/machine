import argparse
import random
import re
import unicodedata
import io

import torch
from torch.autograd import Variable

import Constants

USE_CUDA = False

parser = argparse.ArgumentParser(description='data.py')

# **Preprocess Options**
parser.add_argument('-dataname', type=str, default='SCAN',
                    help="Path to the training data")
parser.add_argument('-trainfile',  type=str,
                    help="Path to the training data")
parser.add_argument('-testfile',  type=str,
                    help="Path to the training data")
parser.add_argument('-savedata',  type=str,
                    help="Output file for the prepared data")

opt = parser.parse_args()

def main():
    print('Preparing training ...')
    vocab, train_pairs = prepare_data(opt.trainfile)
    vocab.trim(Constants.MIN_COUNT)
    train_pairs = clean_pairs(vocab, train_pairs)

    print('Preparing test ...')
    vocab, test_pairs = prepare_data(opt.testfile, vocab)
    vocab.trim(Constants.MIN_COUNT)
    test_pairs = clean_pairs(vocab, test_pairs)

    # TODO: here I need to save the data
    savedata = {'vocab': vocab,
                 'train_pairs': train_pairs,
                 'test_pairs': test_pairs}
    torch.save(savedata, opt.savedata + '.pt')

    # torch.save(vocab, open(opt.savedata + '.vocab.pt', 'wb'))
    # torch.save(train_pairs, open(opt.savedata + '.train.pt', 'wb'))
    # torch.save(test_pairs, open(opt.savedata + '.test.pt', 'wb'))


def prepare_data(filename, vocab=None):
    '''
    The full process for preparing the data is:
    1. Read text file and split into lines
    2. Split lines into pairs and normalize
    3. Filter to pairs of a certain length
    4. Make word lists from sentences in pairs
    '''

    vocab, pairs = read_langs(filename)
    print("Read %d sentence pairs" % len(pairs))

    pairs = filter_pairs(pairs)
    print("Filtered to %d pairs" % len(pairs))

    print("Indexing words...")
    if not vocab:
        vocab = Vocab()
    for pair in pairs:
        vocab.index_words(pair[0])
        vocab.index_words(pair[1])

    print('Indexed %d words in vocab' % (vocab.n_words))
    return vocab, pairs

def clean_pairs(vocab, pairs):
    '''
    Now we will go back to the set of all sentence
    pairs and remove those with unknown words.
    '''
    keep_pairs = []

    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True

        for word in input_sentence.split(' '):
            if word not in vocab.word2index:
                keep_input = False
                break

        for word in output_sentence.split(' '):
            if word not in vocab.word2index:
                keep_output = False
                break

        # Remove if pair doesn't match input and output conditions
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from %d pairs to %d, %.4f of total" % (len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs

class Vocab:
    '''
    We'll need a unique index per word to use as the inputs and targets
    of the networks later. To keep track of all this we will use a helper
    class called Lang which has word => index (word2index) and index => word
    (index2word) dictionaries, as well as a count of each word (word2count).
    This class includes a function trim(min_count) to remove rare words once
    they are all counted.
    '''
    def __init__(self):
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {Constants.PAD_token: "PAD", Constants.SOS_token: "SOS", Constants.EOS_token: "EOS"}
        self.n_words = 3  # Count default tokens

    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed: return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.n_words = 3  # Count default tokens

        for word in keep_words:
            self.index_word(word)

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    '''
    The files are all in Unicode, to simplify we will turn
    Unicode characters to ASCII, make everything lowercase,
    and trim most punctuation.
    '''
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([,.!?])", r" \1 ", s)
    s = re.sub(r"[^a-zA-Z,.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def read_langs(filename):
    '''
    To read the data file we will split the file into lines,
    and then split lines into pairs. The files are all
    source_language => target_language, so if we want to translate
    from target_language => source_language I added the reverse
    flag to reverse the pairs.
    '''

    print("Reading lines...")

    lines = io.open(filename).read().strip().split('\n')
    separator = '\t'
    if opt.dataname == 'SCAN':
        separator = 'OUT:'
        # remove 'IN: ' prefix
        lines = [l[3:] for l in lines]

    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split(separator)] for l in lines]

    return Vocab(), pairs

def filter_pairs(pairs):
    filtered_pairs = []
    for pair in pairs:
        if len(pair[0]) >= Constants.MIN_LENGTH and len(pair[0]) <= Constants.MAX_LENGTH \
            and len(pair[1]) >= Constants.MIN_LENGTH and len(pair[1]) <= Constants.MAX_LENGTH:
                filtered_pairs.append(pair)
    return filtered_pairs

# Return a list of indexes, one for each word in the sentence, plus EOS
def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')] + [Constants.EOS_token]


# Pad a with the PAD symbol
def pad_seq(seq, max_length):
    seq += [Constants.PAD_token for i in range(max_length - len(seq))]
    return seq


def random_batch(batch_size, vocab, pairs):
    input_seqs = []
    target_seqs = []

    # Choose random pairs
    for i in range(batch_size):
        pair = random.choice(pairs)
        input_seqs.append(indexes_from_sentence(vocab, pair[0]))
        target_seqs.append(indexes_from_sentence(vocab, pair[1]))

    # Zip into pairs, sort by length (descending), unzip
    seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs = zip(*seq_pairs)

    # For input and target sequences, get array of lengths and pad with 0s to max length
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
    target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)

    if USE_CUDA:
        input_var = input_var.cuda()
        target_var = target_var.cuda()

    return input_var, input_lengths, target_var, target_lengths

if __name__ == "__main__":
    main()