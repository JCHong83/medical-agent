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
    symptoms: List[str] = Field(description="List of relevant medical symptoms extracted from the conversation")
    emergency_flags: bool = Field(description="True if symptoms indicate an immediate life-threatening emergency")
    is_final: bool = Field(description="True if the user explicitly confirmed they have nothing else to add")
    next_question: str = Field(description="The empathetic follow-up question to ask if is_final is false.")
    specialty_suggestion: str = Field(description="The suggested specialist based on symptoms (e.g., 'Dentist', 'Cardiologist', 'General Practitioner')")

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
            "You are an empathetic medical intake assistant. Your goal is to understand symptoms and identify the correct specialist.\n"
            "LANGUAGE RULES:\n"
            "- You MUST communicate exclusively in English.\n"
            "- If the user described symptoms but hasn't finished, ask: 'I understand. Is there anything else I should know?'\n"
            "- If the user says 'No', 'That is all', or 'It is everything', set is_final=True.\n"
            "- CRITICAL: Identify the specialist in English. Examples: 'Dentist', 'Cardiologist', 'General Practitioner', 'Orthopedist'.\n"
            "- DO NOT use JSON formatting in the response text (next_question)."
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
                "messages": [AIMessage(content="I'm sorry, could you repeat that? I want to make sure I understand your symptoms correctly.")]
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