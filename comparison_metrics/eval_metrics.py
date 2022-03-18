
#From https://towardsdatascience.com/how-to-evaluate-text-generation-models-metrics-for-automatic-evaluation-of-nlp-models-e1c251b04ec1


from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu

import itertools

#import pyter  not able to import atm


'''
read from files - 
ref.txt : reference texts
gen.txt : generated texts (from model)
these files should be in the same directory
'''
def evaluation_metrics(ref_file_path, gen_file_path, n_for_rouge = 2):
    '''
    Args:
        ref_file_path (string) : reference file path -> file containing the reference sentences on each line
        gen_file_path (string) : model generated file path -> containing corresponding generated sentences(to reference sentences) on each line
    
    Returns:
        A list containing [bleu, rouge, meteor, ter]
    '''
    file_ref = open(ref_file_path, 'r')
    ref = file_ref.readlines()

    file_gen = open(gen_file_path, 'r')
    gen = file_gen.readlines()

    for i,l in enumerate(gen):
        gen[i] = l.strip()

    for i,l in enumerate(ref):
        ref[i] = l.strip()
    
    #ter_score = ter(ref, gen)
    bleu_score = bleu(ref, gen)
    rouge_score = rouge_n(ref, gen, n=n_for_rouge)
    return [bleu_score, rouge_score]



#start of Rouge

#rouge scores for a reference/generated sentence pair
#source google seq2seq source code.
#supporting function
def _split_into_words(sentences):
  """Splits multiple sentences into words and flattens the result"""
  return list(itertools.chain(*[_.split(" ") for _ in sentences]))

#supporting function
def _get_word_ngrams(n, sentences):
  """Calculates word n-grams for multiple sentences.
  """
  assert len(sentences) > 0
  assert n > 0

  words = _split_into_words(sentences)
  return _get_ngrams(n, words)

#supporting function
def _get_ngrams(n, text):
  """Calcualtes n-grams.
  Args:
    n: which n-grams to calculate
    text: An array of tokens
  Returns:
    A set of n-grams
  """
  ngram_set = set()
  text_length = len(text)
  max_index_ngram_start = text_length - n
  for i in range(max_index_ngram_start + 1):
    ngram_set.add(tuple(text[i:i + n]))
  return ngram_set

def rouge_n(reference_sentences, evaluated_sentences, n=2):
  """
  Computes ROUGE-N of two text collections of sentences.
  Source: http://research.microsoft.com/en-us/um/people/cyl/download/
  papers/rouge-working-note-v1.3.1.pdf
  Args:
    evaluated_sentences: The sentences that have been picked by the summarizer
    reference_sentences: The sentences from the referene set
    n: Size of ngram.  Defaults to 2.
  Returns:
    recall rouge score(float)
  Raises:
    ValueError: raises exception if a param has len <= 0
  """
  if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
    raise ValueError("Collections must contain at least 1 sentence.")

  evaluated_ngrams = _get_word_ngrams(n, evaluated_sentences)
  reference_ngrams = _get_word_ngrams(n, reference_sentences)
  reference_count = len(reference_ngrams)
  evaluated_count = len(evaluated_ngrams)

  # Gets the overlapping ngrams between evaluated and reference
  overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
  overlapping_count = len(overlapping_ngrams)

  # Handle edge case. This isn't mathematically correct, but it's good enough
  if evaluated_count == 0:
    precision = 0.0
  else:
    precision = overlapping_count / evaluated_count

  if reference_count == 0:
    recall = 0.0
  else:
    recall = overlapping_count / reference_count

  f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))

  #just returning recall count in rouge, useful for our purpose
  return recall


def bleu(ref, gen):
    ''' 
    calculate pair wise bleu score. uses nltk implementation
    Args:
        references : a list of reference sentences 
        candidates : a list of candidate(generated) sentences
    Returns:
        bleu score(float)
    '''
    ref_bleu = []
    gen_bleu = []
    for l in gen:
        gen_bleu.append(l.split())
    for i,l in enumerate(ref):
        ref_bleu.append([l.split()])
    cc = SmoothingFunction()
    score_bleu = corpus_bleu(ref_bleu, gen_bleu, weights=(0, 1, 0, 0), smoothing_function=cc.method4)
    return score_bleu
    
    
'''
    Args:
        ref - reference sentences - in a list
        gen - generated sentences - in a list
    Returns:
        averaged TER score over all sentence pairs
    '''
'''
def ter(ref, gen):

    if len(ref) == 1:
        total_score =  pyter.ter(gen[0].split(), ref[0].split())
    else:
        total_score = 0
        for i in range(len(gen)):
            total_score = total_score + pyter.ter(gen[i].split(), ref[i].split())
        total_score = total_score/len(gen)
    return total_score


'''

if __name__ == "__main__":
    print("\n\n\n'ello\n\n")
    
    referenceList = ["\"Luff, you lubber,\" cried an Irish voice that was Smee\’s; \"here\’s the" , "Colin slowly sat up and stared and stared--as he had stared when he", "Alice thought she saw a way out of the difficulty this time. \"If you\’ll", "\"What!\" shouted Hercules, very wrathfully, \"do you intend to make me", "well-known road, shaking his ears and whisking his tail with a contented"]

    candidateListPhonemes20 = ["Liberty them with a carry with staring after another and their was know doubt the spring and that aye was", "Colin and he walked along to his pockets and sat down and once had not have the. But the", "Alice and he was and they will say their is know. The king which made her so more than.", " Hercules and herself was a sensible matter and came back in the storm of the wall and he was a", " Shaking and the son of some wise. Aye don't know why the princess had been travelled in a day"]
    candidateListWords20 = ["Luff you lubber cried an irish voice that was smees genial little rock. Here is a little brother but there is no other", "Colin slowly sat up in a corner of a street and a few yards farther on the rock was a long time to get up", "Alice thought she saw a little boy in the world and she was in a little boy. He was not very much.", "What shouted hercules very little. But the next morning the earl had heard him and the boy was the most of his people", "Well known road shaking his head. The old man was the only thing in the world but he had not known and the earl"]
    
    candidateListPhonemes35 = ["Liberty a second way together and then he lead him toward her and could not return to the window", "Colin with a bravery men hu were afraid that you was aladdin to send the light and had knocked her and", "Alice to keep him forward to c the last time he went on somewhere and they came to the kingdom", "Hercules self but what he could c how then hu came into their castle his big brother and still was", "Shaking. Aladdin had been afraid that he was still a big little house took with the. The"]
    
    candidateListWords35 = ["Luff you lubber cried a irish voice that was smees heres the rock.", "Colin slowly sat up in his chair and watched her and a great deal of the cottage.He had made his father and", "Alice thought she saw little chap and tell him now. What are you thinking about. It was a very lovely place", "What shouted hercules very wisdom and we will tell you their people. And i have no more trouble to be the same time", "Well known road shaking his head again. But the child had no time for her. I thought i should like to see"]
    
    candidateListCharacters = ["", "", "", "", ""]
    
    
    print("Bleu score is",bleu(referenceList, candidateListPhonemes35))
    print("Rouge-1 score is",rouge_n(referenceList, candidateListPhonemes35, 1))
    print("Rouge-2 score is",rouge_n(referenceList, candidateListPhonemes35, 2))

