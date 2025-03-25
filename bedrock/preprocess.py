import spacy
import nltk
import os
import pytextrank
from transformers import T5Tokenizer, T5ForConditionalGeneration

meeting = open('data/meeting_utk.txt','r').read()

# summarization - may need to be done before the previous two steps 
# or even independent of them
# let's try using a hugging face model in this case

model_name = 't5-large'
tokenizer = T5Tokenizer.from_pretrained(model_name,legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def summarize(text):
    prompt = "Summarize the following meeting minutes in detail, capturing key points, decisions and next steps:" + text
    inputs = tokenizer.encode(prompt, 
                              return_tensors="pt", max_length=1024, 
                              truncation=True)
    summary_ids = model.generate(inputs, max_new_tokens=150,
                                 min_length=50,length_penalty=2.0, 
                                 num_beams=6,early_stopping=True
                                 )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

summary = summarize(meeting)
print(summary)

























nlp = spacy.load('en_core_web_lg')
nlp.add_pipe("textrank")


doc = nlp(meeting)

"""
for sent in doc._.textrank.summary(limit_phrases=5, limit_sentences=3):
    print(sent)
    print('Summary Length - {}'.format(len(sent)))
    1==1
"""

"""
    

# stopword removal
filtered_meeting = [token for token in doc if not token.is_stop]

# lemmatization
filtered_meeting = [token.lemma_ for token in filtered_meeting if token.lemma_]
filtered_meeting = ' '.join(filtered_meeting)

"""