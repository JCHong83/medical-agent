from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.messages import HumanMessage, AIMessage
from app.agents.graph import app as agent_graph
import uvicorn

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

# --- Endpoints ---

@app.post("/chat")
async def chat_with_agent(request: AgentRequest):
  try:
    # 1. Convert incoming messages to LangChain format
    formatted_messages = []
    for msg in request.messages:
      if msg.role == "user":
        formatted_messages.append(HumanMessage(content=msg.content))
      else:
        formatted_messages.append(AIMessage(content=msg.content))
    
    # 2. Prepare the initial state
    initial_state = {
      "messages": formatted_messages,
      "symptoms": [],
      "duration": "",
      "severity": "",
      "emergency_flag": False,
      "specialty_required": "",
      "patient_location": {"lat": request.lat, "lng": request.lng},
      "recommended_doctors": []
    }

    # 3. Run the graph (using ainvole to get the final state)
    final_state = await agent_graph.ainvoke(initial_state)

    # 4. Format the response for the mobile app
    # If emergency, return the emergency message; otherwise return doctor list
    response_text = ""
    if final_state.get("emergency_flag"):
      # Get the message from the emergency node
      response_text = final_state["messages"][-1].content
    else:
      response_text = f"Based on your symptoms, I recommend seeing a {final_state['specialty_required']}."

    return {
      "status": "success",
      "diagnosis_summary": {
        "symptoms": final_state["symptoms"],
        "specialty": final_state["specialty_required"],
        "is_emergency": final_state["emergency_flag"]
      },
      "response_text": response_text,
      "doctors": final_state["recommended_doctors"]
    }
  
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
  
if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8000)