import json,os
import spacy

# while it's impractical in this specific use case, this preprocessing can and will be used elsewhere

nlp = spacy.load('en_core_web_lg')

with open('llms.json','r') as f:
    llms = json.load(f)

def get_llm_json(model_name,llms=llms):
    _similarities = []
    model_name = nlp(model_name)
    for i in range(len(llms)):
        _temp_model_name = llms[i]['modelId']
        _temp_model_name_nlp = nlp(_temp_model_name)
        similarity_score = model_name.similarity(_temp_model_name_nlp)
        _similarities.append(similarity_score)
    return llms[_similarities.index(max(_similarities))]


import fitz
print(fitz)

_doc_ = fitz.open("data/mckinsey_nm.pdf")
text = ""
for page in _doc_:
    text+=page.get_text()
print(text)