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

import pytesseract
from pdf2image import convert_from_path
import cv2
import os
from pypdf import PdfReader



def extract_pdf(file_path):
    reader = PdfReader(file_path)
    full_text = ''
    for page in reader.pages:
        full_text+=page.extract_text()
    return full_text

def read_pdf_cv(file_path):
    pages = convert_from_path(file_path, dpi=300)

    extracted_text = []

    for i, page in enumerate(pages):
    
        image_path = f'temp_page_{i}.png'
        page.save(image_path, 'PNG')

        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        text = pytesseract.image_to_string(thresh, lang='eng')
        extracted_text.append(text)

        os.remove(image_path)

        full_text = '\n'.join(extracted_text)

        print("Done. Extracted text saved to output.txt")
        return full_text

file_path = 'data/mckinsey_nm.pdf'
file_extension = file_path[file_path.index('.'):]
full_text = ''

if file_extension == '.pdf':
    temp_full_text = extract_pdf(file_path)
if full_text == temp_full_text:
    full_text = read_pdf_cv(file_path)
else:
    full_text = temp_full_text
print(full_text)

