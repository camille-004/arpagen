"""Prepare data for neural network."""
import collections
import json
import pickle
from itertools import chain
from typing import List, Tuple

import _io
import nltk
import numpy as np
from nltk.tokenize import word_tokenize

import utils.constants as _constants


def read_phoneme_json(f: _io.TextIOWrapper) -> Tuple[List[List], ...]:
    """Read translated corpus from JSON file and prepare for tokenizing."""
    ph_dict = json.load(f)

    _corpus = []

    for s_num, sent in ph_dict.items():
        curr_sent = []
        for w_num, word in sent.items():
            curr_sent.append(list(word.values()))
        _corpus.append(curr_sent)

    return tuple(_corpus)


def tokenize_sentences(sequence: Tuple[List[List], ...]) -> List:
    """Create sentence list, and append <BOS> and <EOS> tokens."""
    sentences = []

    for sent in sequence:
        for i in range(len(sent)):
            # print(sent[i])
            sent[i] = np.append(sent[i], _constants.SPACE_TOKEN)

        sent_encoding = [[_constants.BOS_TOKEN]] + sent + [[_constants.EOS_TOKEN]]
        sentences.append(list(chain.from_iterable(sent_encoding)))

    sentences = np.array(list(chain.from_iterable(sentences)))

    # Remove <SPACE> before <EOS>
    extra_spaces = np.where(sentences == _constants.EOS_TOKEN)[0] - 1
    sentences = np.delete(sentences, extra_spaces)
    return list(sentences)


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


class Vocab:
    """Vocabulary for text. Supports words, characters, and phonemes."""

    def __init__(self, _tokens=None, min_freq=0, reserved_tokens=None):
        """Initialize phoneme index dictionary and get frequencies."""
        if _tokens is None:
            _tokens = []

        if reserved_tokens is None:
            reserved_tokens = []

        # Sort according to frequencies
        cnt = collections.Counter(list(_tokens))
        self.freqs = sorted(cnt.items(), key=lambda x: x[1], reverse=True)

        # Index for <UNK> is 0
        self.idx_to_tok = [_constants.UNK_TOKEN] + reserved_tokens
        self.tok_to_idx = {token: idx for idx, token in enumerate(self.idx_to_tok)}

        for token, freq in self.frequencies:
            if freq < min_freq:
                self.frequencies[_constants.UNK_TOKEN] += 1
                break
            if token not in self.tok_to_idx:
                self.idx_to_tok.append(token)
                self.tok_to_idx[token] = len(self.idx_to_tok) - 1

    def __len__(self):
        """Get length of vocabulary."""
        return len(self.idx_to_tok)

    def __getitem__(self, _tokens):
        """Extract indices of sentence."""
        if isinstance(_tokens, str):
            return self.tok_to_idx.get(_tokens, self.unk)
        return [self.__getitem__(tok) for tok in _tokens]

    def to_tokens(self, idx):
        """Extract sentence from indices."""
        if isinstance(idx, int):
            return self.idx_to_tok[idx]
        return [self.idx_to_tok[i] for i in idx]

    @property
    def unk(self):
        """Get index of unknown token."""
        return 0

    @property
    def frequencies(self):
        """Get vocabulary frequencies."""
        return self.freqs


class Corpus:
    """Class for corpus from text."""

    def __init__(self, corpus_type: str, f_name: str):
        """Initialize corpus."""
        self.corpus_type = corpus_type
        self.f_name = f_name

        self.tokens = None

    def create(self, subset: int = 0):
        """Get tokens depending on corpus type."""
        file = open(self.f_name)

        if self.corpus_type == "phoneme":
            assert self.f_name.endswith(".json")
            self.tokens = read_phoneme_json(file)

        elif self.corpus_type == "word":
            # Right now, we are assuming the sentences are split by \n characters.
            assert self.f_name.endswith(".txt")
            try:
                self.tokens = [word_tokenize(sent.strip()) for sent in file.readlines()]
            except LookupError:
                nltk.download("punkt")
                self.tokens = [word_tokenize(sent.strip()) for sent in file.readlines()]

        elif self.corpus_type == "char":
            assert self.f_name.endswith(".txt")
            try:
                self.tokens = [
                    [list(w) for w in word_tokenize(sent.strip())]
                    for sent in file.readlines()
                ]
            except LookupError:
                nltk.download("punkt")
                self.tokens = [
                    [list(w) for w in word_tokenize(sent.strip())]
                    for sent in file.readlines()
                ]

        else:
            raise ValueError(
                "corpus_type can only be one of: ['phoneme', 'word', 'char']"
            )

        self.tokens = self.tokens[subset:]
        self.tokens = tokenize_sentences(self.tokens)
        return self.tokens

    def create_vocab(
        self,
        max_tokens: int = -1,
        save: bool = True,
        vocab_save_path: str = None,
        corpus_save_path: str = None,
    ):
        """Encode tokens and create vocabulary.

        Option to save both vocabulary and encoded corpus.
        """
        if self.tokens is None:
            raise AssertionError(
                "Please call create() before instantiating the tokens' vocabulary."
            )

        _vocab = Vocab(
            self.tokens,
            reserved_tokens=[
                _constants.BOS_TOKEN,
                _constants.PADDING_TOKEN,
                _constants.EOS_TOKEN,
            ],
        )

        _corpus = np.array(_vocab[self.tokens])

        if max_tokens > 0:
            _corpus = _corpus[:max_tokens]

        if save:
            if vocab_save_path is None:
                raise AssertionError(
                    "Please specify a path in which to save the vocabulary."
                )

            if corpus_save_path is None:
                raise AssertionError(
                    "Please specify a path in which to save the corpus."
                )

            assert vocab_save_path.endswith(".pkl")
            assert corpus_save_path.endswith(".npy")

            self.save_vocab(_vocab, vocab_save_path)
            np.save(corpus_save_path, _corpus)

        return _corpus, _vocab

    @staticmethod
    def save_vocab(_vocab: Vocab, fp: str):
        """Save a vocab instance model to file path."""
        f = open(fp, "wb")
        pickle.dump(_vocab, f)
        f.close()


def one_hot_encode(_corpus: np.ndarray, _vocab: Vocab) -> np.ndarray:
    """One-hot encode each token in each sentence."""
    one_hot = np.zeros((np.multiply(*_corpus.shape), len(_vocab)), dtype=np.float32)
    one_hot[np.arange(one_hot.shape[0]), _corpus.flatten()] = 1.0
    one_hot = one_hot.reshape((*_corpus.shape, len(_vocab)))

    return one_hot


if __name__ == "__main__":
    corpus = Corpus("phoneme", "../data/phonemes.json")
    print(corpus.create())
    corpus.create_vocab(
        vocab_save_path="../data/ex_vocab_phoneme.pkl",
        corpus_save_path="../data/ex_corpus_phoneme.npy",
    )
