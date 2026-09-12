import json
import os
import boto3
import botocore.exceptions
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

def get_bedrock_client():
    kwargs = {
        "service_name": "bedrock-runtime",
        "region_name": os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    }
    if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
        kwargs["aws_access_key_id"] = os.getenv("AWS_ACCESS_KEY_ID")
        kwargs["aws_secret_access_key"] = os.getenv("AWS_SECRET_ACCESS_KEY")
    if os.getenv("AWS_SESSION_TOKEN"):
        kwargs["aws_session_token"] = os.getenv("AWS_SESSION_TOKEN")
    return boto3.client(**kwargs)

# Model IDs (can be configured via environment variables)
EMBEDDING_MODEL_ID = os.getenv("BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
AMAZON_MODEL_ID = os.getenv("BEDROCK_AMAZON_MODEL", "amazon.nova-lite-v1:0")
LLAMA_MODEL_ID = os.getenv("BEDROCK_LLAMA_MODEL", "meta.llama3-8b-instruct-v1:0")

# Initialize Bedrock client and embeddings
bedrock = get_bedrock_client()
bedrock_embeddings = BedrockEmbeddings(model_id=EMBEDDING_MODEL_ID, client=bedrock)

# Load PDF files and split into chunks
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

# Vector embedding and vector store
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(
        docs,
        embedding=bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")

# Instantiate the LLMs
amazon_llm = ChatBedrock(
    model_id=AMAZON_MODEL_ID,
    client=bedrock,
    model_kwargs={"temperature": 0.7}
)

llama_llm = ChatBedrock(
    model_id=LLAMA_MODEL_ID,
    client=bedrock,
    model_kwargs={"temperature": 0.5}
)

# Prompt template for LLM
prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but summarize with 
at least 250 words and include detailed explanations. If you don't know the answer, 
just say that you don't know; do not try to make up an answer.

<context>
{context}
</context>

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
    answer = qa.invoke({"query": query})
    return answer['result']


import streamlit as st
def main():
    st.set_page_config(page_title="Chat PDF with AWS Bedrock", layout="wide")
    
    st.header("Chat with PDF using AWS Bedrock💁")

    user_question = st.text_input("Ask a Question from the PDF Files")
    
    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                try:
                    docs = data_ingestion()
                    get_vector_store(docs)
                    st.success("Vector store updated successfully")
                except botocore.exceptions.NoCredentialsError:
                    st.error("AWS Credentials not found. Please create a `.env` file with `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` or run `aws configure`.")
                except Exception as e:
                    st.error(f"Error: {e}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Amazon Nova Output", use_container_width=True):
            if not user_question:
                st.warning("Please enter a question first.")
            elif not os.path.exists("faiss_index"):
                st.warning("Please create the vector store first by clicking 'Vectors Update' in the sidebar.")
            else:
                with st.spinner("Processing with Amazon Nova..."):
                    try:
                        faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                        st.write(get_response_llm(amazon_llm, faiss_index, user_question))
                        st.success("Response generated using Amazon Nova")
                    except botocore.exceptions.NoCredentialsError:
                        st.error("AWS Credentials not found. Please configure your AWS credentials.")
                    except Exception as e:
                        st.error(f"Error: {e}")

    with col2:
        if st.button("Meta Llama Output", use_container_width=True):
            if not user_question:
                st.warning("Please enter a question first.")
            elif not os.path.exists("faiss_index"):
                st.warning("Please create the vector store first by clicking 'Vectors Update' in the sidebar.")
            else:
                with st.spinner("Processing with Meta Llama..."):
                    try:
                        faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                        st.write(get_response_llm(llama_llm, faiss_index, user_question))
                        st.success("Response generated using Meta Llama")
                    except botocore.exceptions.NoCredentialsError:
                        st.error("AWS Credentials not found. Please configure your AWS credentials.")
                    except Exception as e:
                        st.error(f"Error: {e}")

    

if __name__ == "__main__":
    main()
