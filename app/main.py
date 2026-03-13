import logging
import time
import os
import shutil
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage

# Google Generative AI for Audio Processing
import google.generativeai as genai

# Import the compiled graph from graph.py
from app.agents.graph import agent_graph
import uvicorn

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MedicalPlusAgent")

# Configure Gemini SDK
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI(title="Medical+ AI Agent API", version="1.0.0")

# --- Security: CORS Middleware ---
# Ensure your mobile app can talk to this server
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"], # In production, replace with your specific domain/IP
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# --- Request/Response Models ---

class ChatMessage(BaseModel):
	role: str # "user" or "assistant"
	content: str

class AgentRequest(BaseModel):
	messages: List[ChatMessage]
	# Defaulting to None forces the mobile app to provide real GPS data
	lat: float
	lng: float
	user_id: Optional[str] = None # For future RLS/Personalization

# --- Helpers ---

async def run_medical_graph(messages, lat, lng, user_id):
	# Helper to execute the LangGraph workflow.
	initial_state = {
		"messages": messages,
		"symptoms": [],
		"duration": "",
		"severity": "moderate",
		"emergency_flag": False,
		"specialty_required": "",
		"patient_location": {"lat": lat, "lng": lng},
		"recommended_doctors": []
	}

	final_state = await agent_graph.ainvoke(initial_state)

	return {
		"status": "success",
		"metadata": {
			"process_id": str(time.time()),
			"is_emergency": final_state.get("emergency_flag", False)
		},
		"diagnosis": {
			"detected_symptoms": final_state.get("symptoms", []),
			"recommended_specialty": final_state.get("specialty_required", "General Practice"),
		},
		"response_text": final_state["messages"][-1].content,
		"doctors": final_state.get("recommended_doctors", [])
	}

# --- Internal Middleware for Timing ---
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
	start_time = time.time()
	response = await call_next(request)
	process_time = time.time() - start_time
	response.headers["X-Process-Time"] = str(process_time)
	return response

# --- Endpoints ---

@app.get("/health")
async def health_check():
	return {"status": "healthy", "timestamp": time.time()}

@app.post("/chat")
async def chat_with_agent(request: AgentRequest):
	formatted_messages = [
		HumanMessage(content=m.content) if m.role == "user" else AIMessage(content=m.content)
		for m in request.messages
	]
	return await run_medical_graph(formatted_messages, request.lat, request.lng, request.user_id)

@app.post("/voice-command")
async def voice_command(
	file: UploadFile = File(...),
	lat: float = Form(...),
	lng: float = Form(...),
	user_id: str = Form(None)
):
	"""
	Receives audio from the React Native Orb, transcribes it,
	and runs the medical graph.
	"""
	logger.info(f"Processing voice command for user: {user_id}")

	# Save temporary audio file
	temp_path = f"temp_{file.filename}"
	with open(temp_path, "wb") as buffer:
		shutil.copyfileobj(file.file, buffer)

	try:
		# Upload to Google File API
		# Note: In production, consider a cleanup strategy for Google Cloud files
		uploaded_file = genai.upload_file(path=temp_path)

		# Use Gemini 2.0 Flash to transcribe with medical context
		model = genai.GenerativeModel("gemini-1.5-flash") # 2.0 Flash also wolrks great here
		prompt = "Transcribe the following medical symptom report exactly as spoken. do not add any diagnosis yet."

		response = model.generate_content([prompt, uploaded_file])
		transcript_text = response.text.strip()

		logger.info(f"Gemini Transcript: {transcript_text}")

		# Feed the transcript into the Medical Graph
		messages = [HumanMessage(content=transcript_text)]
		return await run_medical_graph(messages, lat, lng, user_id)
	
	except Exception as e:
		logger.error(f"Voice Handshake Error: {str(e)}")
		raise HTTPException(status_code=500, detail="The AI failed to hear you clearly.")
	
	finally:
		# Local Cleanup
		if os.path.exists(temp_path):
			os.remove(temp_path)

if __name__ == "__main__":
	uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)