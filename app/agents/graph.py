# IMPORT DEPENDENCIES

from langgraph.graph import StateGraph, START, END
from app.schemas.state import AgentState
from app.agents.intake_agent import IntakeAgent
from app.agents.routing_agent import RoutingAgent
from app.services.maps_service import MapsService

# Initialize our components
intake = IntakeAgent()
router = RoutingAgent()
maps = MapsService()


# DEFINE THE NODES

async def intake_node(state: AgentState):
  # This calls the method we wrote in the previous step
  updates = await intake.call_node(state)
  return updates

async def routing_node(state: AgentState):
  updates = await router.call_node(state)
  return updates

async def search_node(state: AgentState):
  # Simulate getting coordinates from a frontend (Hardcoded for Milan test)
  lat, lng = 45.4642, 9.1900

  # Use the Maps Service to find specialists based on the state
  specialty = state["specialty_required"]
  raw_doctors = maps.find_nearby_doctors(lat, lng, specialty)

  # Add travel time to each doctor
  final_list = []
  for doc in raw_doctors:
    travel = maps.get_travel_info({"lat": lat, "lng": lng}, doc["location"])
    doc.update(travel)
    final_list.append(doc)

  return {"recommended_doctors": final_list}

async def emergency_node(state: AgentState):
  # A dedicated node for urgent cases
  msg ="🚨 URGENT: Your symptoms suggest an emergency. Please call 112/911 or go to the nearest ER immediately."
  return {"messages": [("ai", msg)]}


# DEFINE THE LOGIC (CONDITIONAL EDGES)

def route_decision(state: AgentState):
  if state.get("specialty_required") == "EMERGENCY_SERVICES":
    return "emergency"
  return "search"


# ASSEMBLE THE GRAPH

# 1. Initialize Graph with our STate schema
workflow = StateGraph(AgentState)

# 2. Add Notes
workflow.add_node("intake", intake_node)
workflow.add_node("router", routing_node)
workflow.add_node("search", search_node)
workflow.add_node("emergency", emergency_node)

# 3. Add Edges (The Flow)
workflow.add_edge(START, "intake")
workflow.add_edge("intake", "router")

# 4. Add Conditional Branching
workflow.add_conditional_edges(
  "router",
  route_decision,
  {
    "emergency": "emergency",
    "search": "search"
  }
)

# 5. Finish
workflow.add_edge("search", END)
workflow.add_edge("emergency", END)

# Compile the graph
app = workflow.compile()

