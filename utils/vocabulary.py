"""Defining vocabulary with phonemes."""
import numpy as np

import utils.constants as _constants


class PhonemeVocab:
    """Create a vocabulary of phonemes."""

    def __init__(
        self,
        pad_token=_constants.PADDING_TOKEN,
        eos_token=_constants.EOS_TOKEN,
        unk_token=_constants.UNK_TOKEN,
    ):
        """Initialize phoneme vocabulary class."""
        self.int_to_phoneme = []

        if pad_token is not None:
            self.int_to_phoneme += [pad_token]

        if eos_token is not None:
            self.int_to_phoneme += [eos_token]

        if unk_token is not None:
            self.int_to_phoneme += [unk_token]

        self.phoneme_to_int = {}

    def __call__(self, _phonemes: np.ndarray):
        """Update mapping of phonemes to integers."""
        _phonemes = set(np.array(_phonemes).flatten())
        _phonemes.remove(_constants.PADDING_TOKEN)
        self.int_to_phoneme += list(_phonemes)
        self.phoneme_to_int = {ph: idx for idx, ph in enumerate(self.int_to_phoneme)}


def encode_phonemes(input_phonemes: np.ndarray, _vocab: PhonemeVocab) -> np.ndarray:
    """Replace every phoneme by its integer value defined in the vocabulary."""
    Z = np.zeros(input_phonemes.shape)

    for i, pair in enumerate(input_phonemes):
        for j, sent in enumerate(pair):
            for k, word in enumerate(sent):
                for _l, phoneme in enumerate(word):
                    Z[i, j, k, _l] = _vocab.phoneme_to_int[phoneme]

    return Z
