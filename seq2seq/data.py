import argparse
import random
import re
import unicodedata
import io

import torch
from torch.autograd import Variable

from Vocab import Vocab

def main():
    print('Preparing training ...')
    vocab_source, vocab_target, train_pairs = prepare_data(opt.trainfile)
    # vocab.trim(opt.min_count)
    # train_pairs = clean_pairs(vocab, train_pairs)

    print('Preparing test ...')
    _, _, test_pairs = prepare_data(opt.testfile, vocab_source, vocab_target)
    # vocab.trim(opt.min_count)
    # test_pairs = clean_pairs(vocab, test_pairs)

    # save data
    savedata = {'vocab_source': vocab_source,
                'vocab_target': vocab_target,
                 'train_pairs': train_pairs,
                 'test_pairs': test_pairs}
    torch.save(savedata, opt.savedata + '.pt')

    # torch.save(vocab, open(opt.savedata + '.vocab.pt', 'wb'))
    # torch.save(train_pairs, open(opt.savedata + '.train.pt', 'wb'))
    # torch.save(test_pairs, open(opt.savedata + '.test.pt', 'wb'))


def prepare_data(filename, vocab_source=None, vocab_target=None):
    '''
    The full process for preparing the data is:
    1. Read text file and split into lines
    2. Split lines into pairs and normalize
    3. Filter to pairs of a certain length
    4. Make word lists from sentences in pairs
    '''

    vocab_source, vocab_target, pairs = read_langs(filename)
    print("Read %d sentence pairs" % len(pairs))

    pairs = filter_pairs(pairs)
    print("Filtered to %d pairs" % len(pairs))

    print("Indexing words...")
    if not vocab_source:
        vocab_source = Vocab(pad=opt.pad_token, sos=opt.sos_token, eos=opt.eos_token)
    if not vocab_target:
        vocab_target = Vocab(pad=opt.pad_token, sos=opt.sos_token, eos=opt.eos_token)
    for pair in pairs:
        vocab_source.index_words(pair[0])
        vocab_target.index_words(pair[1])

    print('Indexed %d words in source vocab' % (vocab_source.n_words))
    print('Indexed %d words in target vocab' % (vocab_target.n_words))
    return vocab_source, vocab_target, pairs

# TODO update this function, removed for now as we 
# are not using it in our experiment
# def clean_pairs(vocab, pairs):
#     '''
#     Now we will go back to the set of all sentence
#     pairs and remove those with unknown words.
#     '''
#     keep_pairs = []
# 
#     for pair in pairs:
#         input_sentence = pair[0]
#         output_sentence = pair[1]
#         keep_input = True
#         keep_output = True
# 
#         for word in input_sentence.split(' '):
#             if word not in vocab.word2index:
#                 keep_input = False
#                 break
# 
#         for word in output_sentence.split(' '):
#             if word not in vocab.word2index:
#                 keep_output = False
#                 break
# 
#         # Remove if pair doesn't match input and output conditions
#         if keep_input and keep_output:
#             keep_pairs.append(pair)
# 
#     print("Trimmed from %d pairs to %d, %.4f of total" % (len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
#     return keep_pairs


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
def normalise_string_scan(s):
    #s = unicode_to_ascii(s.lower().strip())
    # s = re.sub(r"([,.!?])", r" \1 ", s)
    # s = re.sub(r"[^a-zA-Z,.!?]+", r" ", s)
    # s = re.sub(r"\s+", r" ", s).strip()
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

    data = io.open(filename)

    separator = '\t'
    if opt.dataname == 'SCAN':
        separator = 'OUT:'

    # take an input line of scan file and return a source-target pair
    def make_source_target(line):
        l = line[3:]   # remove 'IN: ' prefix
        pair = [normalise_string_scan(s) for s in l.split(separator)] # split and normalise
        return pair

    # Split every line into pairs and normalize
    pairs = [make_source_target(line) for line in data]
    
    vocab_source = Vocab(pad=opt.pad_token, sos=opt.sos_token, eos=opt.eos_token)
    vocab_target = Vocab(pad=opt.pad_token, sos=opt.sos_token, eos=opt.eos_token)

    return vocab_source, vocab_target, pairs

def filter_pairs(pairs):
    filtered_pairs = []
    for pair in pairs:
        if len(pair[0]) >= opt.min_length and len(pair[0]) <= opt.max_length \
            and len(pair[1]) >= opt.min_length and len(pair[1]) <= opt.max_length:
                filtered_pairs.append(pair)
    return filtered_pairs

# Return a list of indexes, one for each word in the sentence, plus EOS
def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')] + [opt.eos_token]


# Pad a with the PAD symbol
def pad_seq(seq, max_length):
    seq += [opt.pad_token for i in range(max_length - len(seq))]
    return seq


def random_batch(batch_size, vocab_source, vocab_target, pairs):
    input_seqs = []
    target_seqs = []

    # Choose random pairs
    for i in range(batch_size):
        pair = random.choice(pairs)
        input_seqs.append(indexes_from_sentence(vocab_source, pair[0]))
        target_seqs.append(indexes_from_sentence(vocab_target, pair[1]))

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

    if opt.use_cuda:
        input_var = input_var.cuda()
        target_var = target_var.cuda()

    return input_var, input_lengths, target_var, target_lengths

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data.py')

    # **Preprocess Options**
    parser.add_argument('-dataname', type=str, default='SCAN',
                        help="Path to the training data")
    parser.add_argument('-trainfile',  type=str,
                        help="Path to the training data")
    parser.add_argument('-testfile',  type=str,
                        help="Path to the test data")
    parser.add_argument('-savedata',  type=str,
                        help="Output file for the prepared data")
    parser.add_argument('-pad_token',  type=int, default=0,
                        help="Token used for padding")
    parser.add_argument('-sos_token',  type=int, default=1,
                        help="Start of sentence token")
    parser.add_argument('-eos_token',  type=int, default=2,
                        help="End of sentence token")
    parser.add_argument('-min_length',  type=int, default=3,
                        help="Minimum sentence length")
    parser.add_argument('-max_length',  type=int, default=1000,
                        help="Maximum sentence length")
    parser.add_argument('-min_count',  type=int, default=1,
                        help="Cutoff count for vocabulary")
    parser.add_argument('-use_cuda', action='store_true',
                        help="Set to true to run on GPU")

    opt = parser.parse_args()

    main()
