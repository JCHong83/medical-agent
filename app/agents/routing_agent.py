from langchain_google_genai import ChatGoogleGenerativeAI
from app.schemas.state import AgentState, SpecialtyDecision
from dotenv import load_dotenv

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
    
    # 2. Prepare the prompt for specialty matching
    symptoms_str = ", ".join(state["symptoms"])
    system_prompt = (
      "You are a medical routing expert. Based on the patient's symptoms, "
      "determine the single most appropriate medical specialty. "
      "Common specialties: Cardiology, Dermatology, Orthopedics, "
      "Neurology, Pediatrics, Ophthalmology, Gastroenterology, General Practice."
    )

    user_prompt = f"Symptoms: {symptoms_str}. Duration: {state['duration']}. Severity: {state['severity']}."

    # 3. Get the decision
    decision = await self.structured_llm.ainvoke([
      ("system", system_prompt),
      ("human", user_prompt)
    ])

    # 4. Handle a situatjion where the LLM might still find an emergency
    if decision.is_emergency:
      return {"specialty_required": "EMERGENCY_SERVICES", "emergency_flag": True}
    
    return {"specialty_required": decision.specialty}