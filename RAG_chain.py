from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from huggingface_hub import InferenceClient
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

client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
    token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
)

def ask(question: str, history: list):
    # Format history
    history_text = ""
    for msg in history:
        history_text += f"{msg['role'].capitalize()}: {msg['content']}\n"

    # Retrieve relevant chunks
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    sources = list(set([
        os.path.basename(doc.metadata.get("source", "Unknown")) 
        for doc in docs
    ]))

    # Build prompt
    final_prompt = prompt.format(
        chat_history=history_text,
        context=context,
        question=question
    )

    # Call HF Inference API using chat completion
    response = client.chat_completion(
        messages=[{"role": "user", "content": final_prompt}],
        max_tokens=256,
        temperature=0.3,
    )

    answer = response.choices[0].message.content.strip()
    return {"answer": answer, "sources": sources}