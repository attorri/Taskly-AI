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
    return doc.sents

def _tokenize(doc):
    return [token for token in doc]

def _remove_stopwords(doc):
    return [token for token in doc if not token.is_stop]

def _lemmatization(doc):
    return [token.lemma_ for token in doc if token.lemma_]

def _pos(doc):
    return [[token.text,token.pos_] for token in doc]

def _ner(doc):
    return [[ent.text, ent.label_] for ent in doc.ents]

def main(doc):
    print('Segmentation -\n{}\n\n'.format(_segment(doc)))
    print('Tokenization -\n{}\n\n'.format(_tokenize(doc)))
    print('Stopword Removal -\n{}\n\n'.format(_remove_stopwords(doc)))
    print('Lemmatization - \n{}\n\n'.format(_lemmatization(doc)))
    print('Parts of Speech (POS) - \n{}\n\n'.format(_pos(doc)))
    print('Named Entity Recognition (NER) -\n{}\n\n'.format(_ner(doc)))
    
if __name__ == '__main__':
    main(doc)