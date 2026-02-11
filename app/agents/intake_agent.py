import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from app.schemas.state import AgentState, ExtractedSymptoms

load_dotenv()

class IntakeAgent:
  def __init__(self):
    # Using Google AI Studio's optimized library
    self.llm = ChatGoogleGenerativeAI(
      model="gemini-2.5-flash",
      temperature=0
    )
    self.extractor = self.llm.with_structured_output(ExtractedSymptoms)

  async def call_node(self, state: AgentState):
    """
    This is the actual function the LangGraph node will call.
    """
    # Get the last message from the user
    last_message = state["messages"][-1].content

    system_prompt = (
      "You are a medical triage assistant. Analyze the user's input "
      "to extract symptoms, duration, and severity. Check for emergencies."
    )

    # Invoke structured output
    extracted: ExtractedSymptoms = await self.extractor.ainvoke([
      ("system", system_prompt),
      ("human", last_message)
    ])

    # Return the updates to the state
    return {
      "symptoms": extracted.symptoms,
      "duration": extracted.duration or "unknown",
      "severity": extracted.severity or "unknown",
      "emergency_flag": extracted.emergency_flags
    }