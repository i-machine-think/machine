import random
import re
import socket
import unicodedata

hostname = socket.gethostname()

from seq2seq.masked_cross_entropy import *

import Constants

USE_CUDA = True

# main function
def make_data():
    input_lang, output_lang, pairs = prepare_data('eng', 'fra', True)
    MIN_COUNT = 5
    input_lang.trim(MIN_COUNT)
    output_lang.trim(MIN_COUNT)
    pairs = filter_pairs(input_lang, output_lang, pairs)


def prepare_data(lang1_name, lang2_name, reverse=False):
    '''
    The full process for preparing the data is:
    1. Read text file and split into lines
    2. Split lines into pairs and normalize
    3. Filter to pairs of a certain length
    4. Make word lists from sentences in pairs
    '''

    input_lang, output_lang, pairs = read_langs(lang1_name, lang2_name, reverse)
    print("Read %d sentence pairs" % len(pairs))

    pairs = filter_pairs(pairs)
    print("Filtered to %d pairs" % len(pairs))

    print("Indexing words...")
    for pair in pairs:
        input_lang.index_words(pair[0])
        output_lang.index_words(pair[1])

    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    return input_lang, output_lang, pairs

def filter_pairs(input_lang, output_lang, pairs):
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
            if word not in input_lang.word2index:
                keep_input = False
                break

        for word in output_sentence.split(' '):
            if word not in output_lang.word2index:
                keep_output = False
                break

        # Remove if pair doesn't match input and output conditions
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from %d pairs to %d, %.4f of total" % (len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs

class Lang:
    '''
    We'll need a unique index per word to use as the inputs and targets
    of the networks later. To keep track of all this we will use a helper
    class called Lang which has word → index (word2index) and index → word
    (index2word) dictionaries, as well as a count of each word (word2count).
    This class includes a function trim(min_count) to remove rare words once
    they are all counted.
    '''
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
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

def read_langs(lang1, lang2, reverse=False):
    '''
    To read the data file we will split the file into lines,
    and then split lines into pairs. The files are all
    source_language → target_language, so if we want to translate
    from target_language → source_language I added the reverse
    flag to reverse the pairs.
    '''

    print("Reading lines...")

    # Read the file and split into lines
#     filename = '../data/%s-%s.txt' % (lang1, lang2)
    filename = '../%s-%s.txt' % (lang1, lang2)
    lines = open(filename).read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

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


def random_batch(batch_size, input_lang, output_lang, pairs):
    input_seqs = []
    target_seqs = []

    # Choose random pairs
    for i in range(batch_size):
        pair = random.choice(pairs)
        input_seqs.append(indexes_from_sentence(input_lang, pair[0]))
        target_seqs.append(indexes_from_sentence(output_lang, pair[1]))

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