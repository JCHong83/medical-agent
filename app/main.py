from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional
import os
from google import genai
from google.genai import types
from langchain_core.messages import HumanMessage, AIMessage
from app.agents.graph import app as agent_graph
from dotenv import load_dotenv
import uvicorn
import time

load_dotenv()

# Configure Gemini for Transcription
client = genai.Client(
  api_key=os.getenv("GOOGLE_API_KEY"),
  http_options={'api_version': 'v1'}
)

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
    "duration": "",
    "severity": "",
    "emergency_flag": False,
    "specialty_required": "",
    "patient_location": {"lat": lat, "lng": lng},
    "recommended_doctors": []
  }

  # Run the graph
  final_state = await agent_graph.ainvoke(initial_state)

  # Format unified response
  response_text = final_state["messages"][-1].content

  return {
    "status": "success",
    "metadata": {
      "is_emergency": final_state.get("emergency_flag", False)
    },
    "diagnosis": {
      "detected_symptoms": final_state.get("symptoms", []),
      "recommended_specialty": final_state.get("specialty_required", "General")
    },
    "response_text": response_text,
    "doctors": final_state.get("recommended_doctors", [])
  }


print(f"Key loaded: {os.getenv('GOOGLE_API_KEY')[:5]}...")

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