"""Prepare data for neural network."""
from typing import List, Tuple

import numpy as np

from corpus_composition_tool import functions

PADDING_TOKEN = "<PAD>"


def split_input_target(sequence: Tuple[List[List[str]], ...]) -> np.array:
    """Split sentences into input and ground truth data.

    The input should include all but the last phoneme of the sentence, and the ground
    truth should be the sentences shifted right one phoneme.
    """
    result = []

    for sentence in sequence:
        st = np.zeros(
            [len(sentence), len(max(sentence, key=lambda x: len(x)))], dtype=object
        )
        for i, j in enumerate(sentence):
            st[i][0:j] = j
        st = np.where(st == 0, PADDING_TOKEN, st)
        result.append(st)

    return np.array(result, dtype=object)


if __name__ == "__main__":
    arpabet = functions.get_arpabet()
    text = ["I am testing this out.", "Hi, Camille."]
    phonemes = functions.sentences_to_phonemes(arpabet, text, 1, len(text))
    split_dataset = split_input_target(phonemes)
    print(split_dataset)
