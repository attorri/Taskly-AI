import spacy
import nltk
import os
import pytextrank

nlp = spacy.load('en_core_web_lg')
nlp.add_pipe("textrank")

meeting = open('data/output.txt','r').read()
doc = nlp(meeting)

# summarization - may need to be done before the previous two steps 
# or even independent of them
# let's try using a hugging face model in this case



"""
    

# stopword removal
filtered_meeting = [token for token in doc if not token.is_stop]

# lemmatization
filtered_meeting = [token.lemma_ for token in filtered_meeting if token.lemma_]
filtered_meeting = ' '.join(filtered_meeting)

"""