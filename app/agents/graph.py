# IMPORT DEPENDENCIES

from langgraph.graph import StateGraph, START, END
from app.schemas.state import AgentState
from app.agents.intake_agent import IntakeAgent
from app.agents.routing_agent import RoutingAgent
from app.services.maps_service import MapsService
from langchain_core.messages import AIMessage

# Initialize our components
intake = IntakeAgent()
router = RoutingAgent()
maps = MapsService()


# --- NODES ---

async def intake_node(state: AgentState):
  # Parses symptoms from user input.
  return await intake.call_node(state)

async def routing_node(state: AgentState):
  # Determines required medical specialty.
  return await router.call_node(state)

async def search_node(state: AgentState):
  # Fetches real doctors based on specialty and dynamic GPS coordinates
  # Grab coordinates passed from the frontend via the state
  location = state.get("patient_location", {})
  lat = location.get("lat")
  lng = location.get("lng")
  specialty = state.get("specialty_required", "General Practitioner")
  
  if not lat or not lng:
    return {
      "messages": [AIMessage(content="I couldn't access your location fto find nearby clinics.")]
    }
  
  # Use MapService to get doctors with travel info included
  recommended_doctors = await maps.find_nearby_doctors(
    lat=lat,
    lng=lng,
    specialty=specialty
  )

  summary = f"I've found {len(recommended_doctors)} {specialty} locations near you."
  return {
    "recommended_doctors": recommended_doctors,
    "messages": [AIMessage(content=summary)]
  }

async def emergency_node(state: AgentState):
  # Handles urgent triage bypass.
  msg ="🚨 URGENT: Your symptoms suggest an emergency. Please call local emergency services (112/911) or go to the nearest ER immediately."
  return {"messages": [AIMessage(content=msg)]}


# --- CONDITIONAL LOGIC ---

def route_decision(state: AgentState):
  # Determines if the flow goes to Map Search or Emergency warning.
  if state.get("emergency_flag") or state.get("specialty_required") == "Emergency Room":
    return "emergency"
  return "search"


# --- GRAPH ASSEMBLY ---

workflow = StateGraph(AgentState)

# Add Notes
workflow.add_node("intake", intake_node)
workflow.add_node("router", routing_node)
workflow.add_node("search", search_node)
workflow.add_node("emergency", emergency_node)

# Define Flow
workflow.add_edge(START, "intake")
workflow.add_edge("intake", "router")

workflow.add_conditional_edges(
  "router",
  route_decision,
  {
    "emergency": "emergency",
    "search": "search"
  }
)

workflow.add_edge("search", END)
workflow.add_edge("emergency", END)

# Compile the runnable graph
agent_graph = workflow.compile()

