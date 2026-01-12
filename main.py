import os
import json
import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# --- 1. SETUP & MOCK MODE DETECTION ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Check if we are in "Real" mode or "Free/Mock" mode
if OPENAI_API_KEY:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    print("✅ OpenAI Key found. Running in LIVE mode.")
    USE_MOCK_AI = False
else:
    client = None
    print("⚠️ No OpenAI Key found. Running in MOCK mode (Free).")
    USE_MOCK_AI = True

# Database Setup
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="user_style")

# --- 2. DATA LOADING ---
def load_memory():
    if not os.path.exists("past_chats.json"):
        return

    with open("past_chats.json", "r") as f:
        chats = json.load(f)
    
    if not chats:
        return

    documents = [msg["text"] for msg in chats]
    ids = [str(i) for i in range(len(chats))]
    metadatas = [{"tag": msg.get("tag", "general")} for msg in chats]

    try:
        existing_ids = collection.get()['ids']
        if existing_ids:
            collection.delete(ids=existing_ids)
    except:
        pass

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"✅ Loaded {len(chats)} messages into memory.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_memory()
    yield

app = FastAPI(lifespan=lifespan)

class ReplyRequest(BaseModel):
    incoming_text: str
    sender_name: str
    relationship: str 

# --- 3. THE ENDPOINT ---
@app.post("/generate")
def generate_replies(request: ReplyRequest):
    
    # --- BRANCH A: NO MONEY (MOCK MODE) ---
    if USE_MOCK_AI:
        return {
            "replies": (
                "1. Neutral: [MOCK] Hey, got your message. (Real AI is off)\n"
                "2. Friendly: [MOCK] This is a test reply! 😉\n"
                "3. Direct: [MOCK] Please add API Key to make me real."
            ),
            "status": "mock_mode"
        }

    # --- BRANCH B: REAL AI (ONLY RUNS IF YOU PAID) ---
    try:
        # 1. Retrieve Style
        results = collection.query(
            query_texts=[request.incoming_text], 
            n_results=3
        )
        retrieved_docs = results['documents'][0] if results['documents'] else []
        style_context = "\n".join([f"- {doc}" for doc in retrieved_docs])

        # 2. Prompt
        system_prompt = f"""
        You are acting as the user '{request.sender_name}'.
        Mimic this style:
        {style_context}
        """
        
        user_prompt = f"Incoming: '{request.incoming_text}'. Generate 3 replies."

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        return {"replies": response.choices[0].message.content, "status": "real_ai"}
    
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    mode = "MOCK (Free)" if USE_MOCK_AI else "LIVE (Paid)"
    return {"status": "Kue Backend is Running", "mode": mode}
