import os
import json
import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
    # NEW: Sliders instead of just "Vibe"
    interest_score: int = 50   # 0 to 100
    spice_score: int = 50      # 0 to 100 (or Assertiveness for work)
    custom_input: Optional[str] = None 

# --- 3. LIFESPAN & APP ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
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
        # 1. Memory Retrieval (ChromaDB)
        results = collection.query(query_texts=[request.incoming_text], n_results=3)
        style_docs = results['documents'][0] if results['documents'] else []
        style_context = "\n".join([f"- {s}" for s in style_docs]) if style_docs else "(No past samples found)"

        # 2. Build Strategy
        system_instruction = build_system_prompt(request, style_context)
        
        # 3. Call OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Incoming Message: \"{request.incoming_text}\"\n\nDecode and Generate."}
            ],
            response_format={ "type": "json_object" }, 
            temperature=0.7 
        )
        
        return json.loads(response.choices[0].message.content)

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- 5. PROMPT ENGINEERING (The Logic Core) ---

def build_system_prompt(req: RequestPayload, style_data: str):
    category = req.dossier.category
    i_score = req.interest_score
    s_score = req.spice_score
    
    # 5a. Define Persona & Slider Meaning
    if category == "Work":
        persona = "You are a Corporate Strategist. Safety Rails: ON (No emojis, no slang)."
        slider_context = f"Professionalism Level: {i_score}%, Assertiveness Level: {s_score}%"
        neg_definition = "Polite but Firm Refusal"
        pos_definition = "Agreement with Action Items"
    elif category == "Dating":
        persona = "You are a High-EQ Dating Coach. Safety Rails: OFF."
        slider_context = f"Interest Level: {i_score}% (Simp vs Cool), Spice Level: {s_score}% (Flirty/Risk)"
        neg_definition = "Playful Disinterest / Sassy"
        pos_definition = "High Enthusiasm / Flirting"
    else:
        persona = "You are a Witty Friend."
        slider_context = f"Warmth: {i_score}%, Roast Level: {s_score}%"
        neg_definition = "Roast / Disagreement"
        pos_definition = "Hype / Validation"

    # 5b. Define Generation Task
    if req.custom_input:
        generation_task = f"""
        USER CUSTOM REQUEST: "{req.custom_input}"
        Apply the sliders ({slider_context}) to this request.
        Generate 3 variations.
        keys: ["option_1", "option_2", "option_3"]
        """
    else:
        generation_task = f"""
        Generate 3 Distinct Options based on ({slider_context}):
        1. "positive": {pos_definition}
        2. "neutral": Safe / Buying Time
        3. "negative": {neg_definition}
        """

    # 5c. The Master Prompt
    return f"""
    ROLE: {persona}
    CONTEXT: {slider_context}
    
    INPUT DATA:
    - User is replying to: {req.dossier.name} ({req.dossier.role_title})
    - Memory (Style Samples):
    {style_data}
    
    YOUR TASK (Output JSON Only):
    
    1. "analysis": Perform cognitive analysis.
       - "translation": What they ACTUALLY mean (Subtext).
       - "threat_level": Integer 0-100.
       - "strategy_advice": 1 sentence strategic tip.
       
    2. "replies": {generation_task}
    """

# --- 6. MOCK DATA ---
def generate_mock_response(req: RequestPayload):
    return {
        "analysis": {
            "translation": "Mock Mode: They want attention.", 
            "threat_level": 45, 
            "strategy_advice": "Keep it cool."
        },
        "replies": {
            "positive": f"Sure thing! (Interest: {req.interest_score}%)",
            "neutral": "Maybe later.",
            "negative": f"No thanks. (Spice: {req.spice_score}%)"
        }
    }

# --- 7. HEALTH CHECK ---
@app.get("/")
def home():
    return {"status": "Kue Brain v2 Ready", "mode": "MOCK" if USE_MOCK_AI else "LIVE"}
