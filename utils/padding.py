"""Prepare data for neural network."""
from typing import List, Tuple

import numpy as np

import utils.constants as _constants
from corpus_composition_tool import functions


def pad_sentences(sequence: Tuple[List[List[str]], ...]) -> np.ndarray:
    """Add padding to sentences."""
    result = []

    for sentence in sequence:
        st = np.zeros(
            [len(sentence), len(max(sentence, key=lambda x: len(x)))], dtype=object
        )
        for i, j in enumerate(sentence):
            st[i][0 : len(j)] = j
        st = np.where(st == 0, _constants.PADDING_TOKEN, st)
        result.append(st)

    return np.array(result, dtype=object)


def split_input_target(padded_sentences: np.ndarray) -> np.ndarray:
    """Split sentences into input and ground truth data.

    The input should include all but the last phoneme of the sentence, and the ground
    truth should be the sentences shifted right one phoneme.
    """
    result = []

    for sent in padded_sentences:
        _input = sent.copy()
        last_input_non_pad = np.argwhere(_input != _constants.PADDING_TOKEN)[-1]
        _input[tuple(last_input_non_pad.T)] = _constants.PADDING_TOKEN

        target = sent.copy()
        first_target_non_pad = np.argwhere(target != _constants.PADDING_TOKEN)[0]
        target[tuple(first_target_non_pad.T)] = _constants.PADDING_TOKEN

        result.append(np.array([_input, target]))

    max_phonemes = len((max(result, key=lambda x: x.shape[2]))[0][0])

    padded_result = []

    for data in result:
        pair = []
        for j, sent in enumerate(data):
            _padded = np.pad(sent, (0, max_phonemes - sent.shape[1]))
            for p in _padded:
                if all(v == 0 for v in p):
                    _padded = _padded[:-1]
            pair.append(_padded)
        padded_result.append(pair)

    padded_result = np.array(padded_result, dtype=object)
    padded_result = np.where(
        padded_result == 0, _constants.PADDING_TOKEN, padded_result
    )

    return padded_result


def pad_data(data: np.array) -> np.array:
    """Pad dataset of sentence pairs.

    The sentences will padded to be the same length, i.e., resulting in a perfectly cubic 3-D matrix.
    """
    max_sent_len = max(len(r[0]) for r in data)
    max_phoneme_len = data[0][0].shape[1]
    Z = np.zeros((len(data), 2, max_sent_len, max_phoneme_len), dtype=object)

    for i, pair in enumerate(data):
        for j, sent in enumerate(pair):
            for k, word in enumerate(sent):
                for _l, phoneme in enumerate(word):
                    Z[i, j, k, _l] = phoneme

    Z_pad = np.where(Z == 0, _constants.PADDING_TOKEN, Z)

    return Z_pad


if __name__ == "__main__":
    arpabet = functions.get_arpabet()
    text = ["I am testing this out.", "Hi, Camille.", "Python is hard sometimes!"]
    phonemes = functions.sentences_to_phonemes(arpabet, text, 1, len(text))
    padded = pad_sentences(phonemes)
    data_pair = split_input_target(padded)
    dataset_padded = pad_data(data_pair)
