"""Vanilla phoneme-based RNN."""

import pickle

import numpy as np
from torch import nn

from utils.preprocess import PhonemeVocab


def get_batches(_corpus: np.ndarray, n_seqs: int, n_steps: int):
    """Create a generator that returns batches of n_seqs x n_steps."""
    batch_size = n_seqs * n_steps
    n_batches = _corpus.shape[0] // batch_size

    # Keep enough phonemes to make only full batches
    _corpus = _corpus[: n_batches * batch_size]

    # Reshape into n_seqs rows
    _corpus = _corpus.reshape((n_seqs, -1))

    for i in range(0, _corpus.shape[1], n_steps):
        _X = _corpus[:, i : i + n_steps]
        _y = np.zeros_like(_X)

        try:
            _y[:, :-1], _y[:, -1] = _X[:, 1:], _corpus[:, i + n_steps]
        except IndexError:
            _y[:, :-1], _y[:, -1] = _X[:, 1:], _corpus[:, 0]

        yield _X, _y


class PhonemeRNN(nn.Module):
    """Phoneme-based RNN."""

    def __init__(
        self,
        _tokens: np.ndarray,
        _vocab: PhonemeVocab,
        # n_steps: int = 100,
        n_hidden: int = 256,
        n_layers: int = 2,
        drop_prob: float = 0.5,
        lr: float = 0.001,
    ):
        """Define architecture of RNN."""
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        # Get vocabulary
        self.phonemes = _tokens
        self.int_to_phoneme = _vocab.idx_to_phoneme
        self.phoneme_to_int = _vocab.phoneme_to_idx

        self.lstm = nn.LSTM(
            len(_vocab), n_hidden, n_layers, dropout=drop_prob, batch_first=True
        )
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden, len(_vocab))

        self.init_weights()

    def init_weights(self):
        """Initialize weights for fully connected layer."""
        # Set bias tensor
        self.fc.bias.data.fill_(0)

        # Uniform random FC weights
        self.fc.weight.data.uniform_(-1, 1)


if __name__ == "__main__":
    corpus = np.load("../data/ex_corpus_100.npy")
    vocab = pickle.load(open("../data/ex_vocab_100.pkl", "rb"))
    rnn = PhonemeRNN(corpus, vocab)
