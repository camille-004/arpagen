"""Prepare data for neural network."""
import collections
from itertools import chain
from typing import List, Tuple

import numpy as np

import utils.constants as _constants
from corpus_composition_tool import functions


def tokenize_sentences(phoneme_sequence: Tuple[List[List[str]]]) -> List[np.ndarray]:
    """Create phoneme-wise sentence list, and append <BOS> and <EOS> tokens."""
    sentences = []

    for sent in phoneme_sequence:
        for i in range(len(sent) - 1):
            sent[i] = np.append(sent[i], _constants.SPACE_TOKEN)

        sent_encoding = [[_constants.BOS_TOKEN]] + sent + [[_constants.EOS_TOKEN]]
        sentences.append(np.array(list(chain.from_iterable(sent_encoding))))

    return sentences


def pad_tokenized_sentences(
    sentences: List[np.ndarray], max_len: int = None
) -> np.ndarray:
    """Add <PAD> tokens to make all sentences the same length."""
    if max_len is None:
        max_len = len(max(sentences, key=lambda x: len(x)))

    for i in range(len(sentences)):
        sentences[i] = list(sentences[i]) + [_constants.PADDING_TOKEN] * (
            max_len - len(sentences[i])
        )

    return np.array(sentences)


class PhonemeVocab:
    """Phoneme-based vocabulary for text."""

    def __init__(self, _tokens=None, min_freq=0, reserved_tokens=None):
        """Initialize phoneme index dictionary and get frequencies."""
        if _tokens is None:
            _tokens = []

        if reserved_tokens is None:
            reserved_tokens = []

        # Sort according to frequencies
        cnt = collections.Counter(_tokens.flatten())
        self._ph_freqs = sorted(cnt.items(), key=lambda x: x[1], reverse=True)

        # Index for <UNK> is 0
        self.idx_to_phoneme = [_constants.UNK_TOKEN] + reserved_tokens
        self.phoneme_to_idx = {
            token: idx for idx, token in enumerate(self.idx_to_phoneme)
        }

        for token, freq in self._ph_freqs:
            if freq < min_freq:
                self._ph_freqs[_constants.UNK_TOKEN] += 1
                break
            if token not in self.phoneme_to_idx:
                self.idx_to_phoneme.append(token)
                self.phoneme_to_idx[token] = len(self.idx_to_phoneme) - 1

    def __len__(self):
        """Get length of vocabulary."""
        return len(self.idx_to_phoneme)

    def __getitem__(self, _tokens):
        """Extract indices of sentence."""
        if isinstance(_tokens, str):
            return self.phoneme_to_idx.get(_tokens, self.unk)
        return [self.__getitem__(tok) for tok in _tokens]

    def to_phonemes(self, idx):
        """Extract phoneme sentence from indices."""
        if isinstance(idx, int):
            return self.idx_to_phoneme[idx]
        return [self.idx_to_phoneme[i] for i in idx]

    @property
    def unk(self):
        """Get index of unknown token."""
        return 0

    @property
    def ph_freqs(self):
        """Get vocabulary frequencies."""
        return self._ph_freqs


def get_corpus_and_vocab(f_name: str, max_tokens: int = -1):
    """Retrieve the corpus and vocabulary for text in a given file."""
    arpabet = functions.get_arpabet()
    file = open(f_name)
    data = functions.text_to_sentences(
        file.read(), r"\.|\!|\?|\n(?=[A-Z])", r"[^a-zA-Z ]+", (10, 100)
    )[0]
    data = data[-100:]

    to_phonemes = functions.sentences_to_phonemes(arpabet, data, 1, len(data))

    tokens = tokenize_sentences(to_phonemes)
    tokens = pad_tokenized_sentences(tokens)

    _vocab = PhonemeVocab(
        tokens,
        reserved_tokens=[
            _constants.BOS_TOKEN,
            _constants.PADDING_TOKEN,
            _constants.EOS_TOKEN,
        ],
    )

    _corpus = np.array(_vocab[tokens])

    if max_tokens > 0:
        _corpus = _corpus[:max_tokens]

    return _corpus, _vocab


def one_hot_encode(_corpus: np.ndarray, _vocab: PhonemeVocab) -> np.ndarray:
    """One-hot encode each phoneme in each sentence."""
    one_hot = np.zeros((np.multiply(*_corpus.shape), len(_vocab)), dtype=np.float32)
    one_hot[np.arange(one_hot.shape[0]), _corpus.flatten()] = 1.0
    one_hot = one_hot.reshape((*_corpus.shape, len(_vocab)))

    return one_hot


if __name__ == "__main__":
    corpus, vocab = get_corpus_and_vocab("../" + _constants.BIBLE_TEXT)
    print(corpus.shape)