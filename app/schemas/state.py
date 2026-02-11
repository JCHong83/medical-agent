from typing import TypedDict, List, Annotated, Optional
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# 1. This is for the LLM's internal extraction logic
class ExtractedSymptoms(BaseModel):
  symptoms: List[str] = Field(description="List of physical or mental symptoms.")
  duration: Optional[str] = Field(description="How long the symptoms have been occurring.")
  severity: Optional[str] = Field(description="Qualitative measure (mild, moderate, severe).")
  emergency_flags: bool = Field(description="True if symptoms indicate an immediate life-threatening emergency.")

class SpecialtyDecision(BaseModel):
  specialty: str = Field(description="The medical specialty required (e.g., Cardiology, Dermatology, etc. )")
  reasoning: str = Field(description="Brief explanation of why this specialty was chosen based on symptoms.")
  is_emergency: bool = Field(description="A secondary check to confirm if this requires an ER visit.")

# 2. This is the shared dmemory of your AI Agent
class AgentState(TypedDict):
  # This keeps track of the conversation history
  messages: Annotated[list, add_messages]
  # Specific medical context extracted
  symptoms: List[str]
  duration: str
  severity: str
  emergency_flag: bool
  specialty_required: str
  patient_location: dict
  recommended_doctors: List[dict]