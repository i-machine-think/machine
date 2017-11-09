class Vocab:
    '''
    We'll need a unique index per word to use as the inputs and targets
    of the networks later. To keep track of all this we will use a helper
    class called Lang which has word => index (word2index) and index => word
    (index2word) dictionaries, as well as a count of each word (word2count).
    This class includes a function trim(min_count) to remove rare words once
    they are all counted.
    '''
    def __init__(self, pad=0, sos=1, eos=2):
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {pad: "PAD", sos: "SOS", eos: "EOS"}
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
