import os
from dotenv import load_dotenv
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, SystemMessage
from app.core.state import AgentState

load_dotenv()

# Define the extraction schema
class TriageOutput(BaseModel):
    symptoms: List[str] = Field(description="Lista di sintomi rilevanti estrapolati dalla conversazione")
    emergency_flags: bool = Field(description="Vero se i sintomi indicano pericolo di vita imminente")
    is_final: bool = Field(description="Vero se l'utente ha confermato esplicitamente di non aver altro da aggiungere")
    next_question: str = Field(description="La domanda empatica da porre se is_final è falso.")
    specialty_suggestion: str = Field(description="Lo specialista suggerito basato sui sintomi (es. 'Dentista', 'Cardiologo', 'Medico di base')")

class IntakeAgent:
    def __init__(self):
        # UPDATED: Use 2.0-flash or 2.5-flash as per your environment requirements
        # This ensures we don't hit the 404 error seen in your logs
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0
        )
        self.extractor = self.llm.with_structured_output(TriageOutput)

    async def call_node(self, state: AgentState):
        messages = state.get("messages", [])
        
        system_prompt = (
            "Sei un assistente medico empatico. Il tuo obiettivo è capire i sintomi e identificare lo specialista corretto.\n"
            "REGOLE DI LINGUAGGIO:\n"
            "- Se l'utente ha descritto dei sintomi ma non ha finito, chiedi: 'Capito. C'è qualcos'altro che dovrei sapere?'\n"
            "- Se l'utente dice 'No', 'Basta così' o 'È tutto', imposta is_final=True.\n"
            "- Fondamentale: Identifica lo specialista. Se l'utente ha mal di denti, la specialty_suggestion deve essere 'Dentista'.\n"
            "- NON usare formati JSON nel testo della risposta (next_question)."
        )

        try:
            response: TriageOutput = await self.extractor.ainvoke([
                SystemMessage(content=system_prompt),
                *messages 
            ])
        except Exception as e:
            # This was causing your issue. We'll log it and try a softer fallback.
            print(f"❌ Intake Node Error: {e}")
            return {
                "is_gathering_complete": False, # Don't stop the conversation if we hit a glitch
                "messages": [AIMessage(content="Scusa, potresti ripetere? Sto cercando di capire meglio i tuoi sintomi.")]
            }
        
        # Merge symptoms
        current_symptoms = state.get("symptoms", [])
        new_symptoms = list(set(current_symptoms + response.symptoms))

        # IMPORTANT: We update the specialty_required in the state here
        updates = {
            "symptoms": new_symptoms,
            "emergency_flag": response.emergency_flags,
            "is_gathering_complete": response.is_final,
            "specialty_required": response.specialty_suggestion,
            "messages": [AIMessage(content=response.next_question)]
        }

        if response.is_final:
            print(f"📊 Triage Complete. Specialty: {response.specialty_suggestion}")
        
        return updates