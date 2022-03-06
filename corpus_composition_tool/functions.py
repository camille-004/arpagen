"""Functions for corpus construction tool."""

import json
import re
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize


def get_arpabet() -> Dict[str, List[List[str]]]:
    """Retrieve arpabet from NLTK."""
    try:
        arpabet = nltk.corpus.cmudict.dict()
    except LookupError:
        nltk.download("cmudict")
        arpabet = nltk.corpus.cmudict.dict()

    return arpabet


def word_to_phonetic(arpabet: Dict[str, List[List[str]]], word: str) -> List[str]:
    """Define translation from word to phonetics.

    If word not in cmudict dictionary, find closest match (SequenceMatcher).
    """
    word = word.lower()

    try:
        phonetic = arpabet[word]
    except Exception:
        keys = arpabet.keys()
        how_similar = [SequenceMatcher(None, word, key).ratio() for key in keys]
        max_index = how_similar.index(max(how_similar))
        phonetic = list(arpabet.values())[max_index]

    if type(phonetic) == list:
        phonetic = phonetic[0]

    return phonetic


def phonetic_to_word(arpabet: Dict[str, List[List[str]]], phonemes: List[list]) -> str:
    """Define translation from phonetics to words.

    If word not in cmudict dictionary, find closest match (SequenceMatcher).
    """
    try:
        word = list(arpabet.keys())[list(arpabet.values()).index(phonemes)]
    except Exception:
        phonemes = phonemes[0]
        values = arpabet.values()
        how_similar = [
            SequenceMatcher(None, phonemes, value[0]).ratio() for value in values
        ]
        max_index = how_similar.index(max(how_similar))
        word = list(arpabet.keys())[max_index]

        if type(word) == list:
            word = word[0]

    return word


def text_to_sentences(
    data: str,
    remove_chars: str,
    r: Tuple[int, int],
    to_lower_case: bool = True,
) -> Tuple[List[str], int]:
    """Pre-processing of *.txt into sentences."""
    data = sent_tokenize(data)
    data = [d.replace("\n", " ") for d in data]
    data = [re.sub(remove_chars, "", d) for d in data]
    data = [re.sub(r"http\S+", "", d) for d in data]
    data = [re.sub(r"www\S+", "", d) for d in data]

    if to_lower_case:
        data = [d.lower() for d in data]

    data = [" ".join([i for i in d.split(" ") if i]) for d in data]
    data = [d for d in data if r[0] <= len(d) <= r[1]]
    unique_chars = set(" ".join(data))

    return data, len(unique_chars)


def sentences_to_phonemes(
    arpabet: Dict[str, List[List[str]]],
    data: List,
    print_every: int,
    of: int,
) -> Tuple[List[List[str]], ...]:
    """Convert list of sentences to list of phoneme lists."""
    data = [
        (
            [word_to_phonetic(arpabet, word) for word in d.split(" ")],
            (print("Line:", i, "of", of) if i % print_every == 0 else ""),
        )
        for i, d in enumerate(data, 1)
    ]

    return list(zip(*data))[0]


def phonemes_to_sentences(
    arpabet: Dict[str, List[List[str]]],
    data: Tuple[List[list], ...],
    print_every: int,
    of: int,
) -> Tuple[str, ...]:
    """Convert list of phoneme lists to sentences."""
    data = [
        (
            " ".join([phonetic_to_word(arpabet, [p]) for p in d]),
            (print("Line:", i, "of", of) if i % print_every == 0 else ""),
        )
        for i, d in enumerate(data, 1)
    ]

    return list(zip(*data))[0]


def sentences_to_words(
    data: List,
    print_every: int,
    of: int,
) -> Tuple[List[List[str]], ...]:
    """Convert list of sentences to list of word lists."""
    try:
        data = [
            (
                [word_tokenize(d)],
                (print("Line:", i, "of", of) if i % print_every == 0 else ""),
            )
            for i, d in enumerate(data, 1)
        ]
    except LookupError:
        nltk.download("punkt")
        data = [
            (
                [word_tokenize(d)],
                (print("Line:", i, "of", of) if i % print_every == 0 else ""),
            )
            for i, d in enumerate(data, 1)
        ]

    return list(zip(*data))[0]


def to_json(json_output_name: str, data: str):
    """Output phonemes in JSON format."""
    data_as_dict = {
        "Sentence "
        + str(i + 1): {  # noqa: W503
            "Word "
            + str(l + 1): {  # noqa: W503
                "Phoneme " + str(m + 1): n for (m, n) in enumerate(k)
            }
            for (l, k) in enumerate(j)
        }
        for (i, j) in enumerate(data)
    }

    json_output = open(json_output_name, "w")
    json_output.write(json.dumps(data_as_dict, indent=4))
    json_output.close()


def from_json(json_input_file: str) -> Tuple[List[list], ...]:
    """Get phonemes from JSON format."""
    data = json.load(open(json_input_file))
    data = [[list(n.values()) for n in list(m.values())] for m in list(data.values())]

    return tuple(data)
