import json
import os
import sys
import boto3
import streamlit as st
import warnings
import pytesseract
from pdf2image import convert_from_path
import cv2

## We will be suing Titan Embeddings Model To generate Embedding

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

## Data Ingestion

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.docstore.document import Document

# Vector Embedding And Vector Store

from langchain.vectorstores import FAISS

## LLm Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

## Bedrock Clients
bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)



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




## Data ingestion
def data_ingestion(file_path):
    file_extension = file_path[file_path.index('.'):]
    full_text = ''

    if file_extension == '.pdf':
        temp_full_text = extract_pdf(file_path)
    if full_text == temp_full_text:
        full_text = read_pdf_cv(file_path)
    else:
        full_text = temp_full_text
    doc = Document(page_content=full_text, metadata={"source":file_path})
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents([doc])
    return docs

## Vector Embedding and vector store

def get_vector_store(docs):
    vectorstore_faiss=FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")

def deepseek_r1():
    llm = Bedrock(model_id='deepseek.r1-v1:0', 
                  client=bedrock, 
                  model_kwargs={"inferenceConfig": {"max_tokens": 200}})
    return llm


prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but usse atleast summarize with 
250 words with detailed explaantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response_llm(llm,vectorstore_faiss,query):
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":query})
    return answer['result']


def main():
    st.set_page_config("Chat PDF")
    
    st.header("Chat with PDF using AWS Bedrock💁")

    user_question = st.text_input("Ask a Question from the PDF Files")
    file_path = 'data/mckinsey_nm.pdf'

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion(file_path)
                get_vector_store(docs)
                st.success("Done")

    if st.button("Deepseek Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm=deepseek_r1()
            
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")


if __name__ == "__main__":
    main()


