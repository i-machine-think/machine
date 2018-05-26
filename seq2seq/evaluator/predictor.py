import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Predictor(object):

    def __init__(self, model, src_vocab, tgt_vocab):
        """
        Predictor class to evaluate for a given model.
        Args:
            model (seq2seq.models): trained model. This can be loaded from a checkpoint
                using `seq2seq.util.checkpoint.load`
            src_vocab (seq2seq.dataset.vocabulary.Vocabulary): source sequence vocabulary
            tgt_vocab (seq2seq.dataset.vocabulary.Vocabulary): target sequence vocabulary
        """
        self.model = model.to(device)

        self.model.eval()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab


    def predict(self, src_seq):
        """ Make prediction given `src_seq` as input.

        Args:
            src_seq (list): list of tokens in source language

        Returns:
            tgt_seq (list): list of tokens in target language as predicted
            by the pre-trained model
        """
        src_id_seq = torch.tensor([self.src_vocab.stoi[tok] for tok in src_seq], dtype=torch.long, device=device).view(1, -1)

        softmax_list, _, other = self.model(src_id_seq, [len(src_seq)])
        length = other['length'][0]

        tgt_id_seq = [other['sequence'][di][0].data[0] for di in range(length)]
        tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_id_seq]
        return tgt_seq
