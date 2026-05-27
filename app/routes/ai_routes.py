import os
import json
import time
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from google import genai
from google.genai import types

from app.services.ai_service import run_medical_logic
from app.core.services import tts, maps_service
from app.core.supabase_client import supabase

router = APIRouter(tags=["AI & Search"])

# Initialize Services for Router
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

class ManualSearchRequest(BaseModel):
  specialty: str
  location: Optional[str] = None
  lat: float
  lng: float
  radius: int = 5

@router.post("/voice-command")
async def voice_command(
  file: UploadFile = File(None),
  lat: float = Form(...),
  lng: float = Form(...),
  history: str = Form("[]"),
):
  past_messages = json.loads(history)

  if (file is None or file.filename == "greeting.txt") and len(past_messages) == 0:
    msg = "Hello! I am your MedicalPlus assistant. Please describe your symptoms, and I will help you find the right specialist."
    return {
      "status": "success",
      "response_text": msg,
      "transcript": "",
      "audio": tts.speak(msg),
      "is_gathering_complete": False,
      "doctors": [],
      "diagnosis": {"recommended_specialty": ""}
    }
  
  try:
    transcript = ""
    if file and file.filename != "greeting.txt":
      audio_data = await file.read()
      response = client.models.generate_content(
        model="gemini-3.1-flash-lite", # Directly use preferred model
        contents=[
          "Accurately transcribe the medical symptoms. Return only the text in English.",
          types.Part.from_bytes(data=audio_data, mime_type="audio/mp4")
        ]
      )
      transcript = response.text.strip() if response.text else ""

    result = await run_medical_logic(transcript, lat, lng, past_messages)
    result["transcript"] = transcript
    return result
  except Exception as e:
    print(f"❌ Voice Error: {e}")
    err_msg = "I'm sorry, I had trouble understanding that. Could you please repeat your symptoms?"
    return {"status": "error", "response_text": err_msg, "audio": tts.speak(err_msg), "doctors": []}
  
# --- Endpoint for Manual Search ---
@router.post("/manual-search")
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
  
  pass