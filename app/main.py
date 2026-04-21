from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional
import os
from google import genai
from google.genai import types
from supabase import create_client, Client
from langchain_core.messages import HumanMessage
from app.agents.graph import app as agent_graph
from app.services.maps_service import MapsService
from dotenv import load_dotenv
import uvicorn
import time


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
async def run_medical_logic(text_query: str, lat: float, lng: float):
  # Prepare the initial state with the new message
  initial_state = {
    "messages": [HumanMessage(content=text_query)],
    "symptoms": [],
    "emergency_flag": False,
    "specialty_required": "",
    "patient_location": {"lat": lat, "lng": lng},
    "recommended_doctors": []
  }

  try:
    # Run the graph
    result = await agent_graph.ainvoke(initial_state)
    
    # Safe State Extraction
    # result is in the dictionary representing the final state
    specialty = result.get("specialty_required", "")
    emergency = result.get("emergency_flag", False)
    symptoms = result.get("symptoms", [])
    
    # Determine AI response text
    if result.get("messages"):
      last_msg = result["messages"][-1]
      # Handle both LangChain message objects and strings
      response_text = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
    else:
      response_text = "Ho analizzato i tuoi sintomi."

  except Exception as graph_err:
    print(f"❌ LangGraph Execution Error: {graph_err}")
    # Fallback values so the app doesn't crash
    specialty = ""
    emergency = False
    symptoms = []
    response_text = "Mi dispiace, ho avuto un problema tecnico nell'analisi."


  # Robust Search Term
  # If AI fails to give a specialty, use the transcript itself as search keyword
  search_term = specialty if specialty else text_query
  print(f"DEBUG: Using search term: '{search_term}'")

  # Fetch internal partners (Supabase)
  partners = []
  try:
    # ROBUST MATCHING
    # Look for the specialty in the array
    # Use 'ilike' logic via a computed filter to find partial matches
    res = supabase.table("doctors") \
      .select("id, specialties, profiles!inner(full_name, avatar_url), doctor_clinics(clinics(address))") \
      .eq("verification_status", "verified") \
      .or_(f'specialties.cs.{{"{search_term}"}},bio.ilike.%{search_term}') \
      .execute()
    
    print(f"DEBUG: Supabase query for '{specialty}' returned {len(res.data)} partners.")
    
    for doc in res.data:
      clinics = doc.get("doctor_clinics", [])
      addr = clinics[0]["clinics"]["address"] if clinics else "Milano, Italia"

      partners.append({
        "id": doc["id"],
        "name": doc["profiles"]["full_name"],
        "avatar": doc["profiles"]["avatar_url"],
        "specialization": specialty,
        "rating": 5.0,
        "address": addr,
        "isRegistered": True,
        "distance": "Partner M+",
      })
  except Exception as e:
    print(f"❌ Supabase Fetch Error: {e}")

  # Fetch external results (Google Maps)
  google_results = maps_service.find_nearby_doctors(lat, lng, specialty)

  sanitized_google = []
  for g_doc in google_results:
    g_doc["isRegistered"] = False
    sanitized_google.append(g_doc)

  # Merge Results (Partners first)
  combined_doctors = partners + sanitized_google

  return {
    "status": "success",
    "metadata": {
      "is_emergency": result.get("emergency_flag", False)
    },
    "diagnosis": {
      "detected_symptoms": result.get("symptoms", []),
      "recommended_specialty": specialty
    },
    "response_text": search_term,
    "doctors": combined_doctors
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