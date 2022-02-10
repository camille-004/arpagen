import json
import re
from difflib import SequenceMatcher

import nltk
import numpy as np


def download_arpabet():
    try:
        arpabet = nltk.corpus.cmudict.dict()
    except LookupError:
        nltk.download("cmudict")
        arpabet = nltk.corpus.cmudict.dict()

    return arpabet


def word_to_phonetic(arpabet, word):
    """Define translation from word to phonetics. If word not in cmudict dictionary, find closest match (
    SequenceMatcher)"""
    word = word.lower()

    try:
        phonetic = arpabet[word]
    except:
        keys = arpabet.keys()
        how_similar = [SequenceMatcher(None, word, key).ratio() for key in keys]
        max_index = how_similar.index(max(how_similar))
        phonetic = list(arpabet.values())[max_index]

    if type(phonetic) == list:
        phonetic = phonetic[0]

    return phonetic


def phonetic_to_word(arpabet, phonemes):
    """Define translation from phonetics to words. If word not in cmudict dictionary, find closest match (
    SequenceMatcher)"""
    try:
        word = list(arpabet.keys())[list(arpabet.values()).index(phonemes)]
    except:
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


def text_to_sentences(data, split_str, remove_chars, r, toLowerCase=True):
    """Pre-processing of *.txt into sentences"""
    data = re.split(split_str, data)
    data = [d.replace("\n", " ") for d in data]
    data = [re.sub(remove_chars, "", d) for d in data]

    if toLowerCase:
        data = [d.lower() for d in data]

    data = [" ".join([i for i in d.split(" ") if i]) for d in data]
    data = [d for d in data if r[0] <= len(d) <= r[1]]
    uniqueChars = set(" ".join(data))

    return data, len(uniqueChars)


def sentences_to_phonemes(arpabet, data, print_every, of):
    """Convert list of sentence to list of phoneme lists"""
    data = [
        (
            [word_to_phonetic(arpabet, word) for word in d.split(" ")],
            (print("Line:", i, "of", of) if i % print_every == 0 else ""),
        )
        for i, d in enumerate(data, 1)
    ]

    return list(zip(*data))[0]


def phonemes_to_sentences(arpabet, data, print_every, of):
    """Convert list of phoneme lists to sentences"""
    data = [
        (
            " ".join([phonetic_to_word(arpabet, [p]) for p in d]),
            (print("Line:", i, "of", of) if i % print_every == 0 else ""),
        )
        for i, d in enumerate(data, 1)
    ]

    return list(zip(*data))[0]


def to_json(json_output_name, data):
    """Output phonemes in json format"""
    data_as_dict = {
        "Sentence "
        + str(i + 1): {
            "Word "
            + str(l + 1): {"Phoneme " + str(m + 1): n for (m, n) in enumerate(k)}
            for (l, k) in enumerate(j)
        }
        for (i, j) in enumerate(data)
    }

    json_output = open(json_output_name, "w")
    json_output.write(json.dumps(data_as_dict, indent=4))
    json_output.close()


def from_json(json_input_file):
    """Import phonemes from json format"""
    data = json.load(open(json_input_file))
    data = [[list(n.values()) for n in list(m.values())] for m in list(data.values())]

    return tuple(data)
