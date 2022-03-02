"""Vanilla phoneme-based RNN."""
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from utils.preprocess import PhonemeVocab, one_hot_encode


def get_batches(_corpus: np.ndarray, _n_seqs: int, _n_steps: int):
    """Create a generator that returns batches of n_seqs x n_steps."""
    batch_size = _n_seqs * _n_steps
    n_batches = _corpus.shape[0] // batch_size

    # Keep enough phonemes to make only full batches
    _corpus = _corpus[: n_batches * batch_size]

    # Reshape into n_seqs rows
    _corpus = _corpus.reshape((_n_seqs, -1))

    for i in range(0, _corpus.shape[1], _n_steps):
        _X = _corpus[:, i : i + _n_steps]
        _y = np.zeros_like(_X)

        try:
            _y[:, :-1], _y[:, -1] = _X[:, 1:], _corpus[:, i + _n_steps]
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
        self.vocab = _vocab
        self.int_to_phoneme = _vocab.idx_to_phoneme
        self.phoneme_to_int = _vocab.phoneme_to_idx

        self.lstm = nn.LSTM(
            len(_vocab), n_hidden, n_layers, dropout=drop_prob, batch_first=True
        )
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden, len(_vocab))

        self.init_weights()

    def forward(self, x, hc):
        """Forward pass through the network."""
        # Get x and hidden state (h, c) from LSTM
        #
        # INPUT TO LSTM:
        # _input shape: (batch size, sequence length, input size)
        # hc = (h_0, c_0), h_0 shape: (1 * n_layers, batch size, hidden size (h_in))
        #                             - Initial hidden state
        #                  c_0 shape: (1 * n_layers, batch size, hidden size (h_cell))
        #                             - Initial cell state
        #
        # OUTPUT FROM LSTM:
        # output shape: (batch size, sequence length, 1 * hidden size (h_out))
        # h_n shape: (1 * n_layers, batch size, h_out)
        #   - Final hidden state for each element in batch
        # c_n shape: (1 * n_layers, batch_size, hidden size (cell))
        #   - Final cell state for each element in batch
        x, (h_n, c_n) = self.lstm(x, hc)

        # Pass output through through dropout layer
        # Randomly zero some elements of input tensor with probability self.drop_prob
        # (default 0.5)
        x = self.dropout(x)

        # Stack LSTM cells layers with .view
        # Can now pass in a list of cells and sent output of one cell to the next
        x = x.reshape(x.size()[0] * x.size()[1], self.n_hidden)

        # Put output through fully connected layer
        x = self.fc(x)

        return x, (h_n, c_n)

    def predict(self, phoneme, h=None, cuda=False, top_k=None):
        """Given a phoneme, predict the next phoneme. Return the phoneme and hidden state."""
        if cuda:
            self.cuda()
        else:
            self.cpu()

        if h is None:
            h = self.init_hidden(1)

        x = np.array([[self.phoneme_to_int[phoneme]]])
        x = one_hot_encode(x, self.vocab)

        inputs = torch.from_numpy(x)

        if cuda:
            inputs = inputs.cuda()

        h = tuple([m.data for m in h])

        # Out = score distribution
        out, h = self.forward(inputs, h)

        # Outputs a distribution of next-phoneme scores.
        # Get actual phoneme by applying a softmax function (gives probability
        # distribution) that we can sample to predict the next phoneme.
        p = F.softmax(out, dim=1).data

        if cuda:
            p = p.cpu()

        if top_k is None:
            top_ph = np.arange(len(self.phonemes))
        else:
            p, top_ph = p.topk(top_k)
            top_ph = top_ph.numpy().squeeze()

        p = p.numpy().squeeze()
        phoneme = np.random.choice(top_ph, p=p / p.sum())

        return self.int_to_phoneme[phoneme], h

    def init_weights(self):
        """Initialize weights for fully connected layer."""
        # Set bias tensor
        self.fc.bias.data.fill_(0)

        # Uniform random FC weights
        self.fc.weight.data.uniform_(-1, 1)

    def init_hidden(self, _n_seqs):
        """Create two new tensors (n_layers, n_seqs, n_hidden) for hidden state and cell state of LSTM."""
        weight = next(self.parameters()).data

        return (
            weight.new(self.n_layers, _n_seqs, self.n_hidden).zero_(),
            weight.new(self.n_layers, _n_seqs, self.n_hidden).zero_(),
        )


def train(
    network: PhonemeRNN,
    data: np.ndarray,
    epochs: int = 10,
    _n_seqs: int = 10,
    _n_steps: int = 50,
    lr: int = 0.001,
    clip: int = 5,
    val_frac: int = 0.2,
    cuda: bool = False,
    print_every: int = 10,
):
    """Train RNN."""
    network.train()
    opt = torch.optim.Adam(network.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    val_idx = int(len(data) * (1 - val_frac))
    data, val_data = data[:val_idx], data[val_idx:]

    if cuda:
        network.cuda()

    step = 0

    for i in range(epochs):
        h = network.init_hidden(_n_seqs)
        for x, y in get_batches(data, _n_seqs, _n_steps):
            step += 1

            # One-hot encode, make Torch tensors
            x = one_hot_encode(x, network.vocab)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

            if cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            h = tuple([m.data for m in h])

            network.zero_grad()

            output, h = network.forward(inputs, h)
            loss = criterion(output, targets.view(_n_seqs * _n_steps))
            loss.backward()

            # Avoid exploding gradients
            nn.utils.clip_grad_norm(network.parameters(), clip)

            opt.step()

            if step % print_every == 0:
                # Validation loss
                val_h = network.init_hidden(_n_seqs)
                val_losses = []

                for x, y in get_batches(val_data, _n_seqs, _n_steps):
                    x = one_hot_encode(x, network.vocab)
                    x, y = torch.from_numpy(x), torch.from_numpy(y)

                    val_h = tuple([m.data for m in val_h])

                    inputs, targets = x, y

                    if cuda:
                        inputs, targets = inputs.cuda(), targets.cuda()

                    output, val_h = network.forward(inputs, val_h)
                    val_loss = criterion(output, targets.view(_n_seqs * _n_steps))

                    val_losses.append(val_loss.item())

                print(
                    f"Epoch: {i + 1} / {epochs},",
                    f"Step: {step},",
                    f"Loss: {loss.item():.4f},",
                    "Val Loss: {:.4f}".format(np.mean(val_losses)),
                )


if __name__ == "__main__":
    corpus = np.load("../data/ex_corpus_100.npy")
    vocab = pickle.load(open("../data/ex_vocab_100.pkl", "rb"))

    rnn = PhonemeRNN(corpus, vocab, n_hidden=512, n_layers=2)
    # print(list(get_batches(corpus, 10, 10)))

    n_seqs, n_steps = 4, 4
    train(rnn, corpus, epochs=25, _n_seqs=n_seqs, _n_steps=n_steps)
