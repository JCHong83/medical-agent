from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional
import os
from google import genai
from google.genai import types
from supabase import create_client, Client
from langchain_core.messages import HumanMessage
from app.core.graph import app as agent_graph
from app.services.maps_service import MapsService
from dotenv import load_dotenv
import uvicorn
import time
import re
from app.services.tts_service import TTSService

tts = TTSService()


load_dotenv()

# Configure Gemini for Transcription
client = genai.Client(
  api_key=os.getenv("GOOGLE_API_KEY"),
  http_options={'api_version': 'v1'}
)

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

maps_service = MapsService()
app = FastAPI(title="Medical AI Agent API")

# --- Request/Response Models ---

class ChatMessage(BaseModel):
  role: str # "user" or "assistant"
  content: str

class AgentRequest(BaseModel):
  messages: List[ChatMessage]
  # Optionally allow the frontend to pass current coordinates
  lat: Optional[float] = 45.4642
  lng: Optional[float] = 9.1900

# Helper Function to Run Graph
async def run_medical_logic(text_query: str, lat: float, lng: float, history_messages: list):
  # Prepare the initial state with the new message
  initial_state = {
    "messages": history_messages,
    "symptoms": [],
    "emergency_flag": False,
    "specialty_required": "",
    "patient_location": {"lat": lat, "lng": lng},
    "is_gathering_complete": False # It tracks fi the conversation is done
  }

  final_state = await agent_graph.ainvoke(initial_state)

  # Safe State Extraction
  # result is in the dictionary representing the final state
  specialty = final_state.get("specialty_required", "")
  is_done = final_state.get("is_gathering_complete", False)
  emergency = final_state.get("emergency_flag", False)

  # DEBUG: See what the AI is thinking before we search
  print(f"DEBUG: AI Specialty Decision: '{specialty}'")
  
  # Get the AI's spoken response
  response_text = "Analisi in corso..."
  if final_state.get("messages"):
    last_msg = final_state["messages"][-1]
    response_text = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)

  # Critical Logic: If not done, just return AI text
  if not is_done and not emergency:
    return {
      "status": "success",
      "diagnosis": {"detected_symptoms": final_state.get("symptoms", []), "recommended_specialty": ""},
      "response_text": response_text,
      "doctors": []
    }
  
  # Use the specialty from the AI, fallback to "General Practice" only if truly empty
  specialty_to_search = specialty if (specialty and specialty != "") else "General Practice"

  # Clean the search term: Remove commas, periods, etc.
  clean_specialty = re.sub(r'[^\w\s]', '', specialty_to_search).strip()

  print(f"DEBUG: Final Search Specialty: '{clean_specialty}'")

  # Fetch internal partners (Supabase)
  partners = []
  if not emergency:
    try:
      # Query Supabase using 'ov' (overlap) for the array
      res = supabase.table("doctors") \
        .select("id, specialties, profiles!inner(full_name, avatar_url), doctor_clinics(clinics(address))") \
        .eq("verification_status", "verified") \
        .or_(f'specialties.cs.{{"{clean_specialty}"}},bio.ilike.%{clean_specialty}%') \
        .execute()
      
      
      for doc in res.data:
        clinics = doc.get("doctor_clinics", [])
        addr = clinics[0]["clinics"]["address"] if clinics else "Milano, Italia"

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

  # Fetch external results (Google Maps)
  google_results = maps_service.find_nearby_doctors(lat, lng, clean_specialty)


  for g_doc in google_results:
    g_doc["isRegistered"] = False
    
  # Generate audio for whatever the AI decided to say
  audio_base64 = tts.speak(response_text)

  return {
    "status": "success",
    "metadata": {"is_emergency": emergency},
    "diagnosis": {
      "detected_symptoms": final_state.get("symptoms", []),
      "recommended_specialty": clean_specialty
    },
    "response_text": response_text,
    "audio": audio_base64,
    "doctors": partners + google_results if is_done else [],
  }


# --- AI model discovery ---
def find_available_model():
  print("🔍 Scanning for available Gemini models...")
  try:
    # List all models available to your specific API key
    for m in client.models.list():
      # Check for the modern Flash models first
      # We want models that support 'generateContent'
      if 'generateContent' in m.supported_actions:
        # Prioritize 2.0 or 1.5 Flash
        if "flash" in m.name.lower():
          return m.name.split('/')[-1]
        
  except Exception as e:
    print(f"❌ Scanner Error: {e}")

  return "gemini-1.5-flash" # Safe fallback

ACTIVE_MODEL = find_available_model()
print(f"🚀 Currently Using Model: {ACTIVE_MODEL}")

# --- Endpoints ---

@app.post("/chat")
async def chat_with_agent(request: AgentRequest):
  try:
    last_user_msg = request.messages[-1].content if request.messages else ""
    return await run_medical_logic(last_user_msg, request.lat, request.lng)
  
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
  
@app.post("/voice-command")
async def voice_command(
  file: UploadFile = File(...),
  lat: float = Form(...),
  lng: float = Form(...),
  user_id: str = Form(None)
):
  # Checking if the GPS is working correctly
  print(f"📍 GPS RECEIVED FROM PHONE: {lat}, {lng}")

  try:
    # Read the audio bytes directly
    audio_data = await file.read()

    # Retry logic for 503 errors
    for attempt in range(3):
      try:
        response = client.models.generate_content(
          model=ACTIVE_MODEL,
          contents=[
            "Transcribe the medical symptoms accurately. Return only the text.",
            types.Part.from_bytes(
              data=audio_data,
              mime_type="audio/mp4"
            )
          ]
        )
        break # Success! Exit the loop
      except Exception as e:
        if "503" in str(e) and attempt < 2:
          print(f"⚠️ Server busy (503). Retrying in {attempt + 1}s...")
          time.sleep(attempt + 1)
          continue
        raise e

    transcript = response.text.strip() if response.text else ""
    print(f"✅ Decoded Transcript: {transcript}")
    return await run_medical_logic(response.text, lat, lng)
  

  except Exception as e:
    error_msg = str(e)
    print(f"❌ Error: {error_msg}")

    return {
      "status": "error",
      "metadata": {"is_emergency": False},
      "diagnosis": {"detected_symptoms": [], "recommended_specialty": ""},
      "response_text": f"Sorry, I encountered an error: {error_msg}",
      "doctors": []
    }


@app.get("/health")
async def health():
  return {"status": "healthy"}

if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8000)