from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from RAG_chain import ask

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    history: Optional[List[Message]] = []

@app.post("/api/chat")
def chat(req: ChatRequest):
    history = [{"role": m.role, "content": m.content} for m in req.history]
    result = ask(req.question, history)
    return {
        "answer": result["answer"],
        "sources": [os.path.basename(s) for s in result["sources"]]
    }