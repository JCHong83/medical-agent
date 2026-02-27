import os
from langchain_google_genai import ChatGoogleGenerativeAI
from app.schemas.state import AgentState, SpecialtyDecision
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv

load_dotenv()

class RoutingAgent:
  def __init__(self):
    # Using Gemini 2.0 Pro for the heavy lifting of medical logic
    self.llm = ChatGoogleGenerativeAI(
      model="gemini-2.0-pro",
      temperature=0
    )
    self.structured_llm = self.llm.with_structured_output(SpecialtyDecision)

  async def call_node(self, state: AgentState):
    """
    LangGraph Node: Determines the medical specialty and sets up the search context.
    """
    # Early Exit for emergencies
    # If the intake agent or previous logic already flagged an emergency, we stop here.
    if state.get("emergency_flag"):
      print("[RoutingAgent] Emergency flag detected. Routing to Emergency Services.")
      return {"specialty_required": "Emergency Room"}
    
    # Extract context from State
    symptoms_str = ", ".join(state.get["symptoms", []])
    if not symptoms_str:
      symptoms_str = "No specificc symptoms reported"

    system_prompt = (
      "You are a medical routing expert. Your task is to match symptoms to the "
      "standard medical specialty that should treat them. Use standard names "
      "suitable for a map search (e.g., 'Cardiologist', 'Dermatologist', 'Pediatrician')"
    )

    user_prompt = (
      f"Patient Data:\n"
      f"- Symptoms: {symptoms_str}\n"
      f"- Duration: {state.get('duration', 'unknown')}\n"
      f"- Severity: {state.get('severity', 'moderate')}"
    )

    try:
      # Get Structured Decision
      decision: SpecialtyDecision = await self.structured_llm.ainvoke([
        SystemMessage(content=system_prompt),
        ("human", user_prompt)
      ])

      # Double check for emergency logic wthin this node
      if decision.is_emergency:
        return {
          "specialty_required": "Emergency Room",
          "emergency_flag": True
        }
      
      # Clean up the specialty name for API compatibility
      # Ensure it's a searchable string (e.g., 'Dermatologist' instead of 'Dermatology')
      specialty = decision.specialty.strip()

      print(f"[RoutingAgent] Specialty Decided: {specialty}. Reason: {decision.reasoning}")
      
      return {
        "specialty_required": specialty
      }
    
    except Exception as e:
      print(f"Error in RoutingAgent: {e}")
      # Fallback to General Practice f reasoning fails
      return {"specialty_required": "General Practitioner"}