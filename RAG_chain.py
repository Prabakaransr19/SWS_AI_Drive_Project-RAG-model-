from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from transformers import pipeline
import os

CHROMA_PATH = "chroma_db"

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    return vectorstore

def get_llm():
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

def build_chain():
    vectorstore = load_vectorstore()
    llm = get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )
    return chain