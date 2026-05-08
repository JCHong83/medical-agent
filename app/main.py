import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional
import json
import re
import uvicorn
import time
from google import genai
from google.genai import types
from supabase import create_client, Client
from langchain_core.messages import HumanMessage, AIMessage
from app.core.graph import app as agent_graph
from app.services.maps_service import MapsService
from app.services.tts_service import TTSService
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI(title="Medical AI Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tts = TTSService()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY"),
    http_options={'api_version': 'v1'}
)

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)
maps_service = MapsService()

# --- Helpers ---

def clean_for_speech(text: str) -> str:
    """Removes technical jargon and cleans text for the TTS engine."""
    # Remove markdown, brackets, or JSON-like snippets
    clean = re.sub(r'\{.*\}', '', text)
    clean = re.sub(r'\[.*\]', '', clean)
    clean = clean.replace('*', '').replace('#', '').strip()
    return clean if clean else "How can I help you?"

# --- Request/Response Models ---

class ChatMessage(BaseModel):
    role: str
    content: str

class AgentRequest(BaseModel):
    messages: List[ChatMessage]
    lat: Optional[float] = 45.4642
    lng: Optional[float] = 9.1900

# --- Manual Search Model ---
class ManualSearchRequest(BaseModel):
    specialty: str
    location: Optional[str] = None
    lat: float
    lng: float
    radius: int = 5

# Helper Function to Run Graph
async def run_medical_logic(text_query: str, lat: float, lng: float, past_messages: list):
    formatted_history = []
    for m in past_messages:
        content = m.get('content') or ""
        if m.get('role') == 'user':
            formatted_history.append(HumanMessage(content=content))
        else:
            formatted_history.append(AIMessage(content=content))

    clean_transcript = text_query if text_query else "..."
    formatted_history.append(HumanMessage(content=clean_transcript))

    initial_state = {
        "messages": formatted_history,
        "symptoms": [],
        "emergency_flag": False,
        "specialty_required": "",
        "patient_location": {"lat": lat, "lng": lng},
        "is_gathering_complete": False 
    }

    final_state = await agent_graph.ainvoke(initial_state)

    specialty = final_state.get("specialty_required", "")
    is_done = final_state.get("is_gathering_complete", False)
    emergency = final_state.get("emergency_flag", False)
    
    # IMPROVED: Extract Response Text safely to avoid the "Come posso aiutarti" glitch
    response_text = ""
    if final_state.get("messages"):
        last_msg = final_state["messages"][-1]
        response_text = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
    
    # Fallback if text is empty or too short
    if len(response_text) < 5:
        response_text = "I understand. I am looking for the most suitable specialist for you."

    response_text = clean_for_speech(response_text)

    # Intermediate response (Still gathering info)
    if not is_done and not emergency:
        return {
            "status": "success",
            "diagnosis": {"detected_symptoms": final_state.get("symptoms", []), "recommended_specialty": ""},
            "response_text": response_text,
            "audio": tts.speak(response_text),
            "doctors": []
        }
    
    # We map the Italian output of the AI to your specific DB strings
    specialty_map = {
        "dentist": ["Dentist", "Dentista", "Odontoiatria"],
        "cardiologist": ["Cardiology", "Cardiologo", "Cardiologia"],
        "pediatrician": ["Pediatrics", "Pediatra", "Pediatria"],
        "orthopedist": ["Orthopedics", "Ortopedico", "Ortopedia"],
        "dermatologist": ["Dermatology", "Dermatologo", "Dermatologia"],
        "general practitioner": ["General Practice", "Medico di base", "Medicina Generale"],
        "gynaecologist": ["Gynecology", "Ginecologo", "Ginecologia"]
    }

    # Normalize AI output to find matches in DB
    normalized_input = specialty.lower().strip()
    search_terms = specialty_map.get(normalized_input, [specialty.title() if specialty else "General Practice"])
    
    clean_specialty = re.sub(r'[^\w\s]', '', search_terms[0]).strip()

    print(f"DEBUG: AI said '{specialty}', Searching DB for '{clean_specialty}'")

    partners = []
    if not emergency and is_done:
        try:
            or_filter = ",".join([f'specialties.cs.{{ "{term}" }}' for term in search_terms])
            or_filter += f',bio.ilike.%{clean_specialty}%'
            # DEBUG: Log exact values before query
            print(f"DEBUG: Executing Supabase query for specialty: {clean_specialty}")

            # We search for the mapped English term in your array
            res = supabase.table("doctors") \
                .select("""id, specialties, profiles(full_name, avatar_url), doctor_clinics(clinics(address))""") \
                .eq("verification_status", "verified") \
                .or_(or_filter) \
                .execute()
            
            print(f"DEBUG: Supabase returned {len(res.data)} rows for terms {search_terms}")
            
            for doc in res.data:
                # Log if profiles or clinics are missing (common RLS issue)
                if not doc.get("profiles"):
                    print(f"⚠️ Warning: Doctor {doc['id']} found but 'profiles' is null (RLS?)")
                    continue

                clinics = doc.get("doctor_clinics", [])
                addr = clinics[0]["clinics"]["address"] if clinics and clinics[0].get("clinics") else "Milano, Italia"
                

                partners.append({
                    "id": doc["id"],
                    "name": doc["profiles"]["full_name"],
                    "avatar": doc["profiles"]["avatar_url"],
                    "specialization": clean_specialty,
                    "rating": 5.0,
                    "address": addr,
                    "isRegistered": True,
                    "distance": "Partner M+",
                })
        except Exception as e:
            print(f"❌ Supabase Error: {e}")

    google_results = maps_service.find_nearby_doctors(lat, lng, clean_specialty, radius=5000)
    for g_doc in google_results:
        g_doc["isRegistered"] = False
        
    final_audio = tts.speak(response_text)

    return {
        "status": "success",
        "metadata": {"is_emergency": emergency},
        "diagnosis": {
            "detected_symptoms": final_state.get("symptoms", []),
            "recommended_specialty": clean_specialty
        },
        "response_text": response_text,
        "audio": final_audio,
        "doctors": partners + google_results if is_done else [],
    }

# --- AI model discovery ---
def find_available_model():
    print("🔍 Scanning for available Gemini models...")
    # List of models in order of preference (Newest/Best first)
    preferred_models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    
    try:
        available = [m.name.split('/')[-1] for m in client.models.list() if 'generateContent' in m.supported_actions]
        for model in preferred_models:
            if model in available:
                return model
    except Exception as e:
        print(f"❌ Scanner Error: {e}")

    return "gemini-2.0-flash" # Your baseline working model

ACTIVE_MODEL = find_available_model()

# --- Endpoints ---

@app.post("/voice-command")
async def voice_command(
    file: UploadFile = File(None),
    lat: float = Form(...),
    lng: float = Form(...),
    history: str = Form("[]"),
    user_id: str = Form(None)
):
    past_messages = json.loads(history)
    transcript = "" 

    # Greeting Logic (Centralized)
    is_greeting = (file is None or file.filename == "greeting.txt") and len(past_messages) == 0

    if is_greeting:
        greeting_text = "Hello! I am your MedicalPlus assistant. Please describe your symptoms, and I will help you find the right specialist."
        return {
            "status": "success",
            "response_text": greeting_text,
            "transcript": "",
            "audio": tts.speak(greeting_text),
            "is_gathering_complete": False,
            "doctors": [],
            "diagnosis": {"recommended_specialty": ""}
        }
    
    try:
        if file and file.filename != "greeting.txt":
            audio_data = await file.read()
            for attempt in range(3):
                try:
                    response = client.models.generate_content(
                        model=ACTIVE_MODEL,
                        contents=[
                            "Accurately transcribe the medical symptoms spoken in this audio. Return only the text in English.",
                            types.Part.from_bytes(data=audio_data, mime_type="audio/mp4")
                        ]
                    )
                    transcript = response.text.strip() if response.text else ""
                    break
                except Exception as e:
                    if "503" in str(e) and attempt < 2:
                        time.sleep(attempt + 1)
                        continue
                    raise e

        print(f"✅ Decoded Transcript: {transcript}")
        result = await run_medical_logic(transcript, lat, lng, past_messages)
        result["transcript"] = transcript
        return result

    except Exception as e:
        print(f"❌ Voice Error: {str(e)}")
        fallback_text = "I'm sorry, I had trouble understanding that. Could you please repeat your symptoms?"
        return {
            "status": "error",
            "metadata": {"is_emergency": False},
            "diagnosis": {"detected_symptoms": [], "recommended_specialty": ""},
            "response_text": fallback_text,
            "audio": tts.speak(fallback_text),
            "doctors": []
        }
    

# --- Endpoint for Manual Search ---
@app.post("/manual-search")
async def manual_search(req: ManualSearchRequest):
    try:
        specialty_lookup = {
            "dentist": ["Dentist", "Dentista", "Odontoiatria"],
            "cardiologist": ["Cardiology", "Cardiologo", "Cardiologia"],
            "pediatrician": ["Pediatrics", "Pediatra", "Pediatria"],
            "general practitioner": ["General Practice", "Medico di base"]
        }

        normalized_input = req.specialty.lower().strip()
        search_terms = specialty_lookup.get(normalized_input, [req.specialty.title()])
        primary_term = search_terms[0]

        # Search Supabase
        or_filter = ",".join([f'specialties.cs.{{ "{term}" }}' for term in search_terms])
        res = supabase.table("doctors") \
            .select("id, specialties, profiles(full_name, avatar_url), doctor_clinics(clinics(address))") \
            .eq("verification_status", "verified") \
            .or_(or_filter) \
            .execute()
        
        partners = []
        for doc in res.data:
            clinics = doc.get("doctor_clinics", [])
            addr = clinics[0]["clinics"]["address"] if clinics else "Milano, Italia"
            partners.append({
                "id": doc["id"],
                "name": doc["profiles"]["full_name"],
                "avatar": doc["profiles"]["avatar_url"],
                "specialization": primary_term,
                "rating": 5.0,
                "address": addr,
                "isRegistered": True,
                "distance": "Partner M+"
            })
        
        # 3. Search Google Maps (Pass the radius from the request)
        # Assuming maps_service.find_nearby_doctors accepts a radius param
        google_results = maps_service.find_nearby_doctors(req.lat, req.lng, primary_term, radius=req.radius * 1000)
        
        for g_doc in google_results:
            g_doc["isRegistered"] = False

        return {
            "status": "success",
            "doctors": partners + google_results,
            "diagnosis": {"recommended_specialty": primary_term}
        }
    
    except Exception as e:
        print(f"Manual Search Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)