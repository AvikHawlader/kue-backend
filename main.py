import os
import json
import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from contextlib import asynccontextmanager

# --- CONFIGURATION ---
# We use an environment variable for security. 
# On your laptop, you can hardcode it for testing, but NEVER commit it to GitHub.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 

client = OpenAI(api_key=OPENAI_API_KEY)
chroma_client = chromadb.Client() # In-memory mode for Render Free Tier compatibility
collection = chroma_client.get_or_create_collection(name="user_style")

# --- DATA LOADING (Runs on Startup) ---
def load_memory():
    """Reads JSON and re-teaches the AI your style on startup."""
    if not os.path.exists("past_chats.json"):
        print("⚠️ Warning: past_chats.json not found. AI will be generic.")
        return

    with open("past_chats.json", "r") as f:
        chats = json.load(f)
    
    # Check if data exists to avoid empty errors
    if not chats:
        return

    documents = [msg["text"] for msg in chats]
    ids = [str(i) for i in range(len(chats))]
    metadatas = [{"tag": msg.get("tag", "general")} for msg in chats]

    # Clear old data to prevent duplicates on restart
    try:
        existing_ids = collection.get()['ids']
        if existing_ids:
            collection.delete(ids=existing_ids)
    except:
        pass

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"✅ Loaded {len(chats)} messages into memory.")

# --- LIFESPAN MANAGER (FastAPI specific) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_memory() # Run this when server starts
    yield

app = FastAPI(lifespan=lifespan)

# --- DATA MODELS ---
class ReplyRequest(BaseModel):
    incoming_text: str
    sender_name: str
    relationship: str # e.g., "Boss", "Girlfriend", "Mom"

# --- THE ENDPOINT ---
@app.post("/generate")
def generate_replies(request: ReplyRequest):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API Key is missing on server.")

    # 1. RETRIEVE style examples
    results = collection.query(
        query_texts=[request.incoming_text], 
        n_results=3
    )
    
    # Flatten the list of lists
    retrieved_docs = results['documents'][0] if results['documents'] else []
    style_context = "\n".join([f"- {doc}" for doc in retrieved_docs])

    # 2. CONSTRUCT PROMPT
    system_prompt = f"""
    You are a personal communication assistant acting as the user.
    
    YOUR GOAL:
    Generate 3 distinct replies to the incoming message.
    
    CRITICAL STYLE INSTRUCTIONS:
    - Mimic the tone, capitalization, and emoji usage of the 'USER PAST MESSAGES' below.
    - If the past messages are short/lowercase, be short/lowercase.
    
    USER PAST MESSAGES (STYLE REFERENCE):
    {style_context}
    
    CONTEXT:
    - Message from: {request.sender_name} ({request.relationship})
    """

    user_prompt = f"Incoming Message: '{request.incoming_text}'\n\nGenerate 3 options: [Neutral, Friendly, Direct/Witty]"

    # 3. CALL OPENAI (GPT-4o-mini)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        generated_text = response.choices[0].message.content
        return {"replies": generated_text}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- HEALTH CHECK ---
@app.get("/")
def home():
    return {"status": "Kue Backend is Running", "model": "gpt-4o-mini"}