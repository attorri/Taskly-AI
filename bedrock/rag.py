import json
import os
import sys
import boto3
import streamlit as st

st.set_page_config("Chat PDF")

# Gonna use Titan Embeddings model to to generate embedding
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms.bedrock import Bedrock

# Data Ingestion Libraries
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Convert to Vector Embeddings + Vector Store
from langchain_community.vectorstores import FAISS

## LLM Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


st.header("Chat with PDF using AWS Bedrock")

# AWS Bedrock Clients


bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",client=bedrock)

## Data Ingestion

def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    # Character Text Split - 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000,
                                                   chunk_overlap = 1000)
    docs = text_splitter.split_documents(documents)
    return docs

# Vector Embeddings + Vector Store

def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    # Can save in a db, but for learning purposes we'll save locally
    vectorstore_faiss.save_local("faiss_index")
    



def deepseek_r1(prompt):
    llm = Bedrock(model_id = 'deepseek.r1-v1:0', 
                  client=bedrock, 
                  model_kwargs={
                    "inferenceConfig": {
                    "max_tokens": 200
                    }
                })
    return llm


_prompt_template = """
    Human: Use the following pieces of context to provide the following - 
    Jira Story Name in under 10 words
    Jira Story Description in under 50 words
    5 Subtask names (ONLY the names) - each under 10 words
    <context>
    {context}
    </context>
    Question: {question}
    Assistant:"""

PROMPT = PromptTemplate(
    template=_prompt_template, input_variables=["context","question"]
)

def get_response_llm(llm,vectorstore_faiss,query):
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type="stuff",
        retreiver = vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k":3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt":PROMPT}
    )
    answer = qa({"query":query})
    return answer["result"]


def main():
    user_question = st.text_input("Ask a question from the pdf files")
    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")
        if st.button("Deepseek"):
            with st.spinner("Processing..."):
                faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings)
                llm = deepseek_r1()
                
                # faiss_index
                st.write(get_response_llm(llm,faiss_index,user_question))
                st.success("Done")

if __name__ == "__main__":
    main()