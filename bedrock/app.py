import os
import boto3
import warnings
import pytesseract
from pdf2image import convert_from_path
import cv2
from PyPDF2 import PdfReader
import tiktoken

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# AWS Bedrock Client
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# --- Custom PDF Extraction Functions ---

def extract_pdf(file_path):
    
    reader = PdfReader(file_path)
    full_text = ''
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text
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
    print("OCR extraction complete.")
    return full_text

def data_ingestion(file_path):
    temp_full_text = extract_pdf(file_path)
    if not temp_full_text.strip():
        full_text = read_pdf_cv(file_path)
    else:
        full_text = temp_full_text
    # Wrap the text in a Document object for LangChain
    doc = Document(page_content=full_text, metadata={"source": file_path})
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents([doc])
    return docs

def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")
    return vectorstore_faiss

def deepseek_r1():
    llm = Bedrock(model_id='deepseek.r1-v1:0', 
                  client=bedrock, 
                  model_kwargs={"inferenceConfig": {"max_tokens": 200}})
    return llm

def titan_text():
    llm = Bedrock(
        model_id="amazon.titan-text-lite-v1",
        client=bedrock,
        model_kwargs={
            "maxTokenCount": 1000,
            "stopSequences": [],
            "temperature": 1,
            "topP": 1
        }
    )
    return llm


def trim_text_to_token_limit(text: str, max_tokens: int = 3000, encoding_name: str = "cl100k_base") -> str:
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        return encoding.decode(tokens)
    return text

from langchain.prompts import PromptTemplate

class TruncatingPromptTemplate(PromptTemplate):
    def format(self, **kwargs):
        if "context" in kwargs:
            kwargs["context"] = trim_text_to_token_limit(kwargs["context"], max_tokens=3000)
        return super().format(**kwargs)

prompt_template = """
Human: Use the following pieces of context to generate a Jira ticket.
Provide a Jira Story Name (under 10 words),
a Jira Story Description (under 50 words), and
5 Subtask names (only the names, each under 10 words).
<context>
{context}
</context>
Question: {question}
Assistant:"""

PROMPT = TruncatingPromptTemplate(template=prompt_template, input_variables=["context", "question"])



def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

def main():
    file_path = 'data/mckinsey_nm.pdf'
    index_path = os.path.join("faiss_index", "index.faiss")
    if os.path.exists(index_path):
        print("Loading cached vector store...")
        vectorstore_faiss = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
    else:
        print("Creating vector store from document...")
        docs = data_ingestion(file_path)
        vectorstore_faiss = get_vector_store(docs)
    
    # Define the query to generate a Jira ticket based on the document text.
    question = "Generate a Jira ticket with a story name, description, and subtasks based on the above document."
    llm = titan_text()
    response = get_response_llm(llm, vectorstore_faiss, question)
    print("Jira Ticket Generated:")
    print(response)

if __name__ == "__main__":
    main()
