from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import sqlite3
from datetime import datetime
from RAG_chain import ask

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "chat_history.db"

#DB Setup 
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def get_history(session_id: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT role, content FROM conversations
        WHERE session_id = ?
        ORDER BY timestamp ASC
    """, (session_id,))
    rows = cursor.fetchall()
    conn.close()
    return [{"role": row[0], "content": row[1]} for row in rows]

def save_message(session_id: str, role: str, content: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO conversations (session_id, role, content)
        VALUES (?, ?, ?)
    """, (session_id, role, content))
    conn.commit()
    conn.close()

#Init DB on startup 
init_db()
print("DB ready!")

# Models 
class ChatRequest(BaseModel):
    session_id: str
    question: str

#end points
@app.post("/api/chat")
def chat(req: ChatRequest):
    # 1. Load history from DB
    history = get_history(req.session_id)

    # 2. Save user message to DB
    save_message(req.session_id, "user", req.question)

    # 3. Call RAG chain
    result = ask(req.question, history)

    # 4. Save assistant response to DB
    save_message(req.session_id, "assistant", result["answer"])

    return {
        "answer": result["answer"],
        "sources": [os.path.basename(s) for s in result["sources"]]
    }

@app.get("/api/history/{session_id}")
def get_chat_history(session_id: str):
    history = get_history(session_id)
    return {"session_id": session_id, "history": history}

@app.delete("/api/history/{session_id}")
def clear_history(session_id: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()
    return {"message": "History cleared"}