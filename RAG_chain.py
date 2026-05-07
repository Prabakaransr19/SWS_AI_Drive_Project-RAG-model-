from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from transformers import pipeline

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

def get_llm():
    pipe = pipeline(
        "text2text-generation",
        model="MBZUAI/LaMini-Flan-T5-248M",
        max_new_tokens=256,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

vectorstore = load_vectorstore()
llm = get_llm()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
prompt = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template=PROMPT_TEMPLATE
)

def ask(question: str, history: list):
    # Format history
    history_text = ""
    for msg in history:
        history_text += f"{msg['role'].capitalize()}: {msg['content']}\n"

    # Retrieve relevant chunks
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))

    # Build prompt
    final_prompt = prompt.format(
        chat_history=history_text,
        context=context,
        question=question
    )

    # Generate answer
    answer = llm.invoke(final_prompt)

    return {"answer": answer, "sources": sources}