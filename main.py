import os
import json
import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # <--- IMPORT THIS
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from typing import List, Optional

# --- 1. CONFIGURATION ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    USE_MOCK_AI = False
    print("🚀 SYSTEM: LIVE MODE (Real AI Active)")
else:
    client = None
    USE_MOCK_AI = True
    print("⚠️ SYSTEM: MOCK MODE (Demo Data Active)")

# --- CHROMADB SETUP ---
# Note: On Render (free tier), this DB resets on restart. 
# For persistence, you need a persistent disk or a cloud vector DB.
chroma_client = chromadb.Client() 
collection = chroma_client.get_or_create_collection(name="user_style")

# --- 2. DATA MODELS ---
class Dossier(BaseModel):
    name: str  
    category: str = Field(..., description="Options: 'Work', 'Dating', 'Friends', 'Family'") 
    role_title: str 
    archetype: Optional[str] = "Standard" 

class RequestPayload(BaseModel):
    incoming_text: str
    dossier: Dossier
    custom_input: Optional[str] = None 
    previous_rejects: List[str] = [] 

# --- 3. LIFESPAN & APP ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(lifespan=lifespan)

# --- ⚠️ CRITICAL FIX: CORS MIDDLEWARE ⚠️ ---
# This allows your React app to talk to this Python server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (Change this to your frontend URL in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 4. THE MASTERMIND ENGINE ---

@app.post("/mastermind")
def run_mastermind(request: RequestPayload):
    
    # --- A. MOCK MODE ---
    if USE_MOCK_AI:
        return generate_mock_response(request)

    # --- B. LIVE MODE ---
    try:
        # 1. Retrieve Style Context (ChromaDB)
        results = collection.query(query_texts=[request.incoming_text], n_results=3)
        style_docs = results['documents'][0] if results['documents'] else []
        style_context = "\n".join([f"- {s}" for s in style_docs]) if style_docs else "(No past samples, use standard tone)"

        # 2. Build Strategy
        system_instruction = build_system_prompt(request, style_context)
        
        # 3. Call OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Incoming Message: \"{request.incoming_text}\"\n\nAnalyze and Generate."}
            ],
            response_format={ "type": "json_object" }, 
            temperature=0.7 
        )
        
        return json.loads(response.choices[0].message.content)

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- 5. PROMPT ENGINEERING ---

def build_system_prompt(req: RequestPayload, style_data: str):
    category = req.dossier.category
    
    # 5a. Define Persona
    if category == "Work":
        persona = "You are a Strategic Corporate Communication Expert."
        neg_definition = "Firm Refusal / Setting Boundaries"
        pos_definition = "Agreement / Action-Oriented"
    elif category == "Dating":
        persona = "You are a High-EQ Dating Coach."
        neg_definition = "Disinterest / Sassy"
        pos_definition = "Warmth / Flirting"
    else:
        persona = "You are a witty friend."
        neg_definition = "Disagreement / Roast"
        pos_definition = "Hype / Love"

    # 5b. Define Generation Logic
    if req.custom_input:
        generation_task = f"""
        USER CUSTOM REQUEST: "{req.custom_input}"
        Generate 3 variations of this specific custom request.
        keys: ["option_1", "option_2", "option_3"]
        """
    else:
        generation_task = f"""
        Generate 3 Distinct Sentiment Options:
        1. "positive": {pos_definition}
        2. "neutral": Safe / Non-committal
        3. "negative": {neg_definition}
        """

    return f"""
    ROLE: {persona}
    
    INPUT CONTEXT:
    - User is replying to: {req.dossier.name} ({req.dossier.role_title})
    - User's Style Samples (from Vector DB):
    {style_data}
    
    YOUR TASK (Output JSON Only):
    1. "analysis": Decode the incoming message (translation, threat_level, strategy_advice).
    2. "replies": {generation_task}
    """

# --- 6. MOCK DATA ---
def generate_mock_response(req: RequestPayload):
    category = req.dossier.category
    if req.custom_input:
        return {
            "analysis": {"translation": "Custom Mode", "strategy_advice": "Executing command."},
            "replies": {
                "option_1": f"Custom: {req.custom_input}",
                "option_2": f"Variation: {req.custom_input}",
                "option_3": f"Short: {req.custom_input}"
            }
        }
    
    # Standard Mock
    replies = {
        "positive": "Sounds good!",
        "neutral": "I'll let you know.",
        "negative": "No thanks."
    }
    if category == "Work":
        replies = {
            "positive": "I will handle this immediately.",
            "neutral": "Received, reviewing now.",
            "negative": "My bandwidth is full."
        }
    
    return {
        "analysis": {"translation": "Mock Logic", "strategy_advice": "This is a demo response."},
        "replies": replies
    }

# --- 7. HEALTH CHECK ---
@app.get("/")
def home():
    return {"status": "Kue Mastermind Ready", "mode": "MOCK" if USE_MOCK_AI else "LIVE"}
