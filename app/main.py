from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import time
import logging

from langchain_core.messages import HumanMessage, AIMessage
from app.agents.graph import app as agent_graph
import uvicorn

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MedicalPlusAgent")

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
    lat: float = Field(..., example=45.4642) 
    lng: float = Field(..., example=9.1900)
    user_id: Optional[str] = None # For future RLS/Personalization

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
    logger.info(f"Received request from user: {request.user_id}")
    
    try:
        # 1. Convert incoming messages to LangChain format
        formatted_messages = []
        for msg in request.messages:
            if msg.role == "user":
                formatted_messages.append(HumanMessage(content=msg.content))
            else:
                formatted_messages.append(AIMessage(content=msg.content))
        
        # 2. Prepare the initial state
        # We include more keys for future-proofing (severity, history)
        initial_state = {
            "messages": formatted_messages,
            "symptoms": [],
            "duration": "",
            "severity": "mild", # Default value
            "emergency_flag": False,
            "specialty_required": "",
            "patient_location": {"lat": request.lat, "lng": request.lng},
            "recommended_doctors": []
        }

        # 3. Run the LangGraph orchestration
        # Note: Using config={"configurable": {"thread_id": request.user_id}} 
        # allows LangGraph to persist memory across calls if you use a checkpointer.
        final_state = await agent_graph.ainvoke(initial_state)

        # 4. Extract logic for response
        last_ai_message = final_state["messages"][-1].content if final_state["messages"] else ""
        
        # Handle cases where the graph might fail to find a specialty
        specialty = final_state.get("specialty_required", "General Practice")
        
        # 5. Return structured response
        return {
            "status": "success",
            "metadata": {
                "process_id": str(time.time()),
                "is_emergency": final_state.get("emergency_flag", False)
            },
            "diagnosis": {
                "detected_symptoms": final_state.get("symptoms", []),
                "recommended_specialty": specialty,
            },
            "response_text": last_ai_message, # This is what the TTS will read
            "doctors": final_state.get("recommended_doctors", [])
        }
  
    except Exception as e:
        logger.error(f"Graph Execution Error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="The AI Agent encountered an error processing your symptoms."
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)