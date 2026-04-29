import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.state import AgentState, SpecialtyDecision

load_dotenv()

class RoutingAgent:
  def __init__(self):
    # Using Gemini1.5 Pro for better clinical reasoning
    self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
    self.structured_llm = self.llm.with_structured_output(SpecialtyDecision)

  async def call_node(self, state: AgentState):
    # 1. Check if the Intake Agent alredy flagged an emergency
    if state.get("emergency_flag"):
      return {"specialty_required": "EMERGENCY_SERVICES"}
    
    # Strict specialty mapping
    allowed_specialties = [
      "General Practice",
      "Pediatrics",
      "Dentist",
      "Dermatology",
      "Cardiology",
      "Orthopedics",
      "Gynecology",
      "Ophtalmology"
    ]
    
    # 2. Prepare the prompt for specialty matching
    
    system_prompt = (
      "Sei un esperto di triage. Devi categorizzare l'utente in una specialità.\n"
      "REGOLE ASSOLUTE:\n"
      "- Se appare la parola 'denti', 'dentista', 'gengive', 'carie' -> DEVI scegliere 'Dentist'.\n"
      "- Se appare 'figlio', 'bambino', 'neonato' -> DEVI scegliere 'Pediatrics'.\n"
      "- Se l'utente dice solo 'È tutto' o 'No', guarda i messaggi PRECEDENTI per capire di cosa parlava.\n"
      "NON USARE 'General Practice' se è stato menzionato un sintomo specifico come il mal di denti."
    )

    # Use the entire message history to better understand context
    messages = state.get("messages", [])

    # 3. Get the decision
    try:
      decision: SpecialtyDecision = await self.structured_llm.ainvoke([
        ("system", system_prompt),
        *messages # Pass on history for better reasoning
      ])
    except Exception as e:
      print(f"❌ Routing Error: {e}")
      return {"specialty_required": "General Practice"}

    # 4. Handle a situatjion where the LLM might still find an emergency
    if decision.is_emergency:
      return {"specialty_required": "EMERGENCY_SERVICES", "emergency_flag": True}
    
    # return {"specialty_required": decision.specialty}
    final_specialty = decision.specialty if decision.specialty in allowed_specialties else "General Practice"

    return {"specialty_required": final_specialty}