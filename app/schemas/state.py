from typing import TypedDict, List, Annotated, Optional
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# --- Extraction Schemas (For LLM Structured Output) ---

class ExtractedSymptoms(BaseModel):
  symptoms: List[str] = Field(deafult_factory=list, description="List of physical or mental symptoms.")
  duration: Optional[str] = Field(None, description="How long the symptoms have been occurring.")
  severity: Optional[str] = Field(None, description="Qualitative measure (mild, moderate, severe).")
  emergency_flags: bool = Field(False, description="True if symptoms indicate an immediate life-threatening emergency.")

class SpecialtyDecision(BaseModel):
  specialty: str = Field(description="The medical specialty required (e.g., Cardiology, Dermatology, etc. )")
  reasoning: str = Field(description="Brief explanation of why this specialty was chosen based on symptoms.")
  is_emergency: bool = Field(description="Confirm if this requires an ER visit.")


# --- LangGraph State (The Shared Memory) ---

class AgentState(TypedDict):
  # Standard LangGraph conversation history
  messages: Annotated[list, add_messages]

  # Medical Context (Populated by intake_agent)
  symptoms: List[str]
  duration: Optional[str]
  severity: Optional[str]
  emergency_flag: bool

  # Routing Context (Populated by routing_agent)
  specialty_required: Optional[str]

  # Location Data (Passed from the Mobile App)
  # Expected format: {"lat": float, "lng": float}
  patient_location: Optional[dict]

  # Final Output (The real-world data from Maps Service)
  recommended_doctors: List[dict]

  # Metadata for the frontend
  process_id: str