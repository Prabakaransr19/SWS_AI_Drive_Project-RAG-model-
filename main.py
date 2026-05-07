from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from RAG_chain import build_chain
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load chain once at startup
print("Loading RAG chain...")
chain = build_chain()
print("RAG chain ready!")

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    history: Optional[List[Message]] = []

@app.post("/api/chat")
def chat(req: ChatRequest):
    # Convert history to list of tuples (user, assistant)
    chat_history = []
    messages = req.history
    for i in range(0, len(messages) - 1, 2):
        user_msg = messages[i].content
        ai_msg = messages[i+1].content if i+1 < len(messages) else ""
        chat_history.append((user_msg, ai_msg))

    result = chain.invoke({
        "question": req.question,
        "chat_history": chat_history
    })

    sources = list(set([
        os.path.basename(doc.metadata.get("source", "Unknown"))
        for doc in result["source_documents"]
    ]))

    return {
        "answer": result["answer"],
        "sources": sources
    }