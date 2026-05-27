import re
from langchain_core.messages import HumanMessage, AIMessage
from app.core.supabase_client import supabase
from app.core.graph import app as agent_graph
from app.core.services import maps_service, tts


def clean_for_speech(text: str) -> str:
  """Removes technical jargon and cleans text for the TTS engine."""
  clean = re.sub(r'\{.*\}', '', text)
  clean = re.sub(r'\[.*\]', '', clean)
  clean = clean.replace('*', '').replace('#', '').strip()
  return clean if clean else "How can I help you?"

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

  response_text = ""
  if final_state.get("messages"):
    last_msg = final_state["messages"][-1]
    response_text = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)

  if len(response_text) < 5:
    response_text = "I understand. I am looking for the most suitable specialist for you."
  
  response_text = clean_for_speech(response_text)

  if not is_done and not emergency:
    return {
      "status": "success",
      "diagnosis": {"detected_symptoms": final_state.get("symptoms", []), "recommended_specialty": ""},
      "response_text": response_text,
      "audio": tts.speak(response_text),
      "doctors": []
    }
  
  specialty_map = {
    "dentist": ["Dentist", "Dentista", "Odontoiatria"],
    "cardiologist": ["Cardiology", "Cardiologo", "Cardiologia"],
    "pediatrician": ["Pediatrics", "Pediatra", "Pediatria"],
    "orthopedist": ["Orthopedics", "Ortopedico", "Ortopedia"],
    "dermatologist": ["Dermatology", "Dermatologo", "Dermatologia"],
    "general practitioner": ["General Practice", "Medico di Base", "Medicina Generale"],
    "gynaecologist": ["Gynecology", "Ginecologo", "Ginecologia"]
  }

  normalized_input = specialty.lower().strip()
  search_terms = specialty_map.get(normalized_input, [specialty.title() if specialty else "General Practice"])
  clean_specialty = re.sub(r'[^\w\s]', '', search_terms[0]).strip()

  partners = []
  if not emergency and is_done:
    try:
      or_filter = ",".join([f'specialties.cs.{{ "{term}" }}' for term in search_terms])
      or_filter += f',bio.ilike.%{clean_specialty}%'
      res = supabase.table("doctors") \
        .select("""id, specialties, profiles(full_name, avatar_url), doctor_clinics(clinics(address))""") \
        .eq("verification_status", "verified") \
        .or_(or_filter) \
        .execute()
      
      for doc in res.data:
        if not doc.get("profiles"): continue
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

  return {
    "status": "success",
    "metadata": {"is_emergency": emergency},
    "diagnosis": {
      "detected_symptoms": final_state.get("symptoms", []),
      "recommended_specialty": clean_specialty
    },
    "response_text": response_text,
    "audio": tts.speak(response_text),
    "doctors": partners + google_results if is_done else [],
  }