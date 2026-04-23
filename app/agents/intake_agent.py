import os
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, SystemMessage
from app.core.state import AgentState


# Define a more advanced extraction schema
class TriageOutput(BaseModel):
  symptoms: List[str] = Field(description="Lista di sintomi rilevanti")
  emergency_flags: bool = Field(description="Vero se i sintomi indicano pericolo di vita")
  is_final: bool = Field(description="Vero se l'utente ha confermato di non aver altro da aggiungere")
  next_question: str = Field(description="La domanda da porre all'utente se is_final e' falso")

class IntakeAgent:
  def __init__(self):
    self.llm = ChatGoogleGenerativeAI(
      model="gemini-2.5-flash",
      temperature=0
    )
    self.extractor = self.llm.with_structured_output(TriageOutput)

  async def call_node(self, state: AgentState):
    messages = state.get("messages", [])

    # --- PHASE 1: GREETING ---
    # If no messages exist, the user just opened the agent
    if not messages:
      greeting = "Ciao! Sono il tuo assistente Medical+. Come posso aiutarti oggi?"
      return {
        "messages": [AIMessage(content=greeting)],
        "is_gathering_complete": False
      }
    
    # --- PHASE 2: EXTRACTION & GATHERING ---
    system_prompt = (
      "Sei un assistente medico empatico. Analizza l'input dell'utente per estrarre i sintomi."
      "REGOLE DI CONVERSAZIONE:\n"
      "1. Se l'utente descrive sintomi ma non ha ancora detto che e' tutto, imposta is_final=False"
      "e chiedi 'C'è altro che dovrei sapere?' in modo gentile. \n"
      "2. Se l'utente dice 'No', 'È tutto', 'Basta così' o frasi simili, imposta is_final=True.\n"
      "3. Se rilevi emergenze (dolore toracico, difficoltà respiratoria grave), imposta emergency_flags=True."
    )

    # We pass the conversation history to the extractor
    try:
      response: TriageOutput = await self.extractor.ainvoke([
        SystemMessage(content=system_prompt),
        *messages # This provides context (memory)
      ])
    except Exception as e:
      print(f"❌ Intake Error: {e}")
      return {"is_gathering_complete": True} # Fallback to prevent loops
    
    # --- PHASE 3: STATE UPDATE ---
    updates = {
      "symptoms": list(set(state.get("symptoms", []) + response.symptoms)),
      "emergency_flag": response.emergency_flags,
      "is_gathering_complete": response.is_final
    }

    # If we aren't done, add the AI's "C'è altro?" to the messages
    if not response.is_final:
      updates["messages"] = [AIMessage(content=response.next_question)]
    else:
      updates["messages"] = [AIMessage(content="Perfetto. Analizzo i tuoi sintomi e cerco lo specialista adatto...")]
    
    return updates