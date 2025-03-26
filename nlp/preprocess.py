import spacy
import nltk
from nltk.tokenize import sent_tokenize
import os
import pytextrank
from transformers import T5Tokenizer, T5ForConditionalGeneration
from collections import Counter
from string import punctuation


nltk.download('punkt_tab')

def sents_nltk():
    _sents = sent_tokenize('I am so happy to leave IBM. Fuck them. WeWork+CVS+EMT :))')
    return _sents

nlp = spacy.load('en_core_web_lg')

def get_hotwords(doc):
    obj = {}
    pos_tag = ['PROPN', 'ADJ', 'NOUN']
    for token in doc:
        if token.is_stop or token.text in punctuation:
            continue
        if token.pos_ in pos_tag:
            if token.text not in obj:
                obj[token.text]=0
            obj[token.text]+=1
    return obj


def predict_category(hotwords, doc, topics = ['Healthcare','Politics','Consulting', 'Technology']):
    str_ents = list(doc.ents)
    str_ents = [ent.text for ent in str_ents]
    
    


"""
    

# stopword removal
filtered_meeting = [token for token in doc if not token.is_stop]

# lemmatization
filtered_meeting = [token.lemma_ for token in filtered_meeting if token.lemma_]
filtered_meeting = ' '.join(filtered_meeting)

"""

model_name = 't5-large'
tokenizer = T5Tokenizer.from_pretrained(model_name,legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def summarize(text,_title,_hotwords,_type_of_text="meeting"):
    prompt = f"Summarize the following text, in the context that is from a {_type_of_text}, with the name of such being '{_title}'."
    prompt += f"Also consider that the following are the top {len(_hotwords)} hot words - {_hotwords}': {text}"
    inputs = tokenizer.encode(prompt, 
                              return_tensors="pt", max_length=1024, 
                              truncation=True)
    summary_ids = model.generate(inputs, max_new_tokens=200,
                                 min_length=50,length_penalty=2.0, 
                                 num_beams=6,early_stopping=True
                                 )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def main():
    _sents = sents_nltk()
    print(_sents) 
    
    meeting = open('data/meeting_utk.txt','r').read()
    doc = nlp(meeting)
    hotwords = get_hotwords(doc)
    hotwords = sorted(hotwords.items(), key=lambda x: x[1], reverse=True)[:10]
    hotwords = [hotwords[i][0] for i in range(len(hotwords))]
    print(hotwords)

    _title = "Address by President Obama to the 71st Session of the United Nations General Assembly"
    _type_of_text = "Speech"
    summary = summarize(meeting,_title,hotwords, _type_of_text)
    print(summary)
    
if __name__ == '__main__':
    main()