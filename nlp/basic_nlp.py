"""
Steps:
    1. Segmentation
    2. Tokenization
    3. Remove Stop words
    4. Lemmatization -> 'Inflected' = Derived (from lemma); considers context for root
    5. Parts of Speech (POS) Tagging (VERB, NOUN, ADJ, PROPN,...)
    6. Name Entity Recognition (NER)

"""

import spacy
import json
import pandas as pd

nlp = spacy.load('en_core_web_lg')
doc = open('data/meeting_utk.txt','r').read()
doc = nlp(doc)

def _segment(doc):
    return [sent for sent in doc.sents]

def _tokenize(doc):
    return [token for token in doc]

def _remove_stopwords(doc):
    return [token for token in doc if not token.is_stop and not token.is_punct]

def _lemmatization(doc):
    return [token.lemma_ for token in doc if token.lemma_]

def _pos(doc):
    return [(token.text,token.pos_) for token in doc]

def _ner(doc):
    return [(ent.text, ent.label_) for ent in doc.ents]

def _pipeline(_sents):
    _words = []
    _sents_original = _sents[:]
    for i in range(len(_sents)):
        _words.append(_tokenize(_sents[i]))
        _sents[i] = _remove_stopwords(_sents[i])
        _sents[i] = _lemmatization(_sents[i])
        # refactoring the sentence, as lemma returns strings.
        _sents[i] = nlp(' '.join(_sents[i]))
        _sents[i] = _pos(_sents[i])
        _sents[i] = list(dict.fromkeys(_sents[i]))
    for i in range(len(_sents)):
        _sents[i]+=_ner(_sents_original[i])
    return _sents

def main(doc):
    _sents = _segment(doc)
    fin = _pipeline(_sents)
    print(fin)
    return fin
    
        
    
if __name__ == '__main__':
    main(doc)
    