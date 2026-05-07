from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI # <-- NEW ENGINE
import os

CHROMA_PATH = "chroma_db"

PROMPT_TEMPLATE = """You are an HR assistant for SWS AI.
Answer ONLY from the context below.
If the answer is not in the context, say "I don't have that information in the company documents."

Chat History:
{chat_history}

Context:
{context}

Question: {question}

Answer:"""

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    return vectorstore

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
prompt = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template=PROMPT_TEMPLATE
)

# Initialize the Gemini Model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", # <-- The current generation engine
    temperature=0.3,
    google_api_key=os.environ.get("GOOGLE_API_KEY") 
)

def ask(question: str, history: list):
    # Format history from DB into readable text
    history_text = ""
    for msg in history:
        history_text += f"{msg['role'].capitalize()}: {msg['content']}\n"

    # Retrieve relevant chunks from Chroma
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    sources = list(set([
        doc.metadata.get("source", "Unknown")
        for doc in docs
    ]))

    # Build final prompt
    final_prompt = prompt.format(
        chat_history=history_text,
        context=context,
        question=question
    )

    # Call Gemini API via LangChain
    response = llm.invoke(final_prompt)
    answer = response.content.strip()

    # Return raw sources — main.py handles basename
    return {"answer": answer, "sources": sources}