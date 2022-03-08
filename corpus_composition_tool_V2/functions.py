import json
import re
import nltk
from tqdm.auto import tqdm
from nltk.util import ngrams
from nltk.metrics.distance import jaccard_distance

from typing import Dict, List, Tuple


def get_arpabet() -> Dict[str, List[List[str]]]:
    """Retrieve arpabet from NLTK."""
    try:
        arpabet = nltk.corpus.cmudict.dict()
    except LookupError:
        nltk.download('cmudict')
        arpabet = nltk.corpus.cmudict.dict()

    #Keeping only the first set of phonemes
    for phoneme in arpabet: arpabet[phoneme] = arpabet[phoneme][0]

    return arpabet


def arpabet_recompiler(
    data: str,
    arpabet: Dict[str, List[list]],
    export_file: str='arpabet.pkl'
) -> Dict[str, List[List[str]]]:
    """recompiles arpabet by removing unused words"""
    
    #Data preprocessing
    data = data.replace('\n', ' ')
    data = re.sub(r'[^a-z\' ]+', '', data.lower())
    data = re.split(' ', data)
    data = [" ".join(d.split()) for d in data if len(d) != 0]
    accepted_words = list(arpabet.keys())
    
    #Removing all unused arpabet words
    arpabet_set, data_set = set(arpabet), set(data)
    for word in tqdm(data_set - arpabet_set):
        closest = [(jaccard_distance(set(ngrams(word, 2)),set(ngrams(w, 2))),w) for w in accepted_words if w[0]==word[0]]
        data_set.add(min(closest, key = lambda d: d[0])[1])
    for k in arpabet_set - data_set:
        del arpabet[k]

    #Export new arpabet dictionary
    with open('arpabet.json', 'w') as outfile:
        json.dump(arpabet, outfile)
    

    return arpabet


def arpabet_reader(self, import_file: str) -> Dict[str, List[List[str]]]:
    """Updates arpabet using imported pickle file"""
    with open('arpabet.json') as in_file:
        arpabet = json.load(in_file)
    return arpabet, list(arpabet.keys()), list(arpabet.values())


def to_json(json_output_name: str, data: str):
    """Output phonemes in JSON format."""
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


def from_json(json_input_file: str) -> Tuple[List[list], ...]:
    """Get phonemes from JSON format."""
    data = json.load(open(json_input_file))
    data = [[list(n.values()) for n in list(m.values())] for m in list(data.values())]

    return data


class CorpusTool:

    def __init__(self, arpabet=nltk.corpus.cmudict.dict()):
        self.arpabet = arpabet
        self.accepted_words = list(arpabet.keys())
        self.accepted_phonemes = list(arpabet.values())


    def word_to_phonetic(self, word: str) -> List[str]:
        """
        Translates word into phonetics.

        If word is not found in cmudict dictionary, find closest match (Jaccard Distance)
        """
        try:
            phonetic = self.arpabet[word]
        except:
            closest = [(jaccard_distance(set(ngrams(word, 2)),set(ngrams(w, 2))),w) for w in self.accepted_words if w[0]==word[0]]
            phonetic = self.arpabet[min(closest, key = lambda d: d[0])[1]]
        return phonetic


    def phonetic_to_word(self, phonemes: List[list]) -> str:
        """
        Translates phonetics into words.

        If word is not found in cmudict dictionary, find closest match (Jaccard Distance)
        """
        try:
            word = accepted_words[accepted_phonemes.index(phonemes)]
        except:
            closest = [(jaccard_distance(set(ngrams(phonemes, 2)),set(ngrams(w, 2))),w) for w in accepted_phonemes]
            word = accepted_words[accepted_phonemes.index(min(closest, key = lambda t: t[0])[1])]
        return word


    def sentences_to_phonemes(self, data: List[str]) -> List[list]:
        """Breaks data into sentences to be processed into phonetic"""
        return [self.sentence_to_phonemes(sentence) for sentence in tqdm(data)]

    def sentence_to_phonemes(self, sentence: str) -> List[list]:
        """Breaks sentences into words to be processed into phonetic"""
        return [self.word_to_phonetic(word) for word in sentence.split(' ')]

    def phonemes_to_sentence(self, data: List[list]) -> str:
        """Breaks data into sentences to be processed into words"""
        return [self.phoneme_to_sentence(sentence) for sentence in tqdm(data)]

    def phoneme_to_sentence(self, phonemes: List[list]) -> str:
        """Breaks sentences into phonemes to be processed into words"""
        return ' '.join([self.phonetic_to_word(phoneme) for phoneme in phonemes])


    def text_to_sentences(
        self, 
        data: str, 
        split_str: str='\.|\!|\?', 
        remove_chars: str=r'[^a-z\' ]+'
    ) -> List[str]:
        """"Pre-processing of *.txt into sentences."""
        data = re.split(split_str, data)
        data = [d.replace('\n', ' ') for d in data]
        data = [re.sub(remove_chars, '', d.lower()).lstrip() for d in data]
        data = [re.sub(' +', ' ', d) for d in data]
        return data

    def arpabet_reader(self, import_file: str) -> Dict[str, List[List[str]]]:
        """Updates arpabet using imported pickle file"""
        with open('arpabet.json') as in_file:
            arpabet = json.load(in_file)
        self.arpabet, self.accepted_words, self.accepted_phonemes = arpabet, list(arpabet.keys()), list(arpabet.values())
        return arpabet, list(arpabet.keys()), list(arpabet.values())
    
    def RNN_out_to_str(self, input_string: str) -> str:
        """Converts RNN output to a readable string"""
        sentences = input_string.replace('<BOS>', '').split('<EOS>')
        sentences = [[self.phonetic_to_word(l.split()) for l in n.split('<SPACE>')] for n in sentences]
        sentences = '. '.join([' '.join(i).capitalize() for i in sentences])
        return sentences