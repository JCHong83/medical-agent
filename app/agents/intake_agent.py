import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from app.schemas.state import AgentState, ExtractedSymptoms
from langchain_core.messages import SystemMessage

load_dotenv()

class IntakeAgent:
  def __init__(self):
    # Initializing Gemini 2.5 Flash with structured output capability
    self.llm = ChatGoogleGenerativeAI(
      model="gemini-2.0-flash",
      temperature=0
    )
    self.extractor = self.llm.with_structured_output(ExtractedSymptoms)

  async def call_node(self, state: AgentState):
    """
    LangGraph Node: Extracts clinical data from the latest conversation turn.
    """
    # Grab the last message (this will br our voice transcript)
    last_message = state["messages"][-1].content

    system_prompt = (
      "You are an expert medical intake assistant. Your goal is to parse "
      "the user's description of their condition into structured data. "
      "Be precise with symptoms. If the user mentions chest pain, difficulty "
      "breathing, or severe bleeding, set emergency_flags to True immediately."
    )

    try:
      # Invoke the structured extractor
      extracted: ExtractedSymptoms = await self.extractor.ainvoke([
        SystemMessage(content=system_prompt),
        ("human", last_message)
      ])

      # Defensive mapping to update AgentState
      # We provide fallback values so the Routing Agent doesn't crash on 'None
      return {
        "symptoms": extracted.symptoms if extracted.symptoms else [],
        "duration": extracted.duration if extracted.duration else "not specified",
        "severity": extracted.severity if extracted.severity else "moderate",
        "emergency_flag": extracted.emergency_flags
      }
    
    except Exception as e:
      print(f"Error in IntakeAgent: {e}")
      # Fallback state if the LLM fails to parse
      return {
        "symptoms": [],
        "emergency_flag": False
      }
    