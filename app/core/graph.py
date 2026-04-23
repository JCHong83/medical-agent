
from langgraph.graph import StateGraph, START, END
from app.core.state import AgentState
from app.agents.intake_agent import IntakeAgent
from app.agents.routing_agent import RoutingAgent


# Initialize our components
intake = IntakeAgent()
router = RoutingAgent()


# DEFINE THE NODES

async def intake_node(state: AgentState):
  # Gathers and summarizes symptoms
  return await intake.call_node(state)

async def routing_node(state: AgentState):
  # Determines if it's an emergency and identifies the specialty
  return await router.call_node(state)


async def emergency_node(state: AgentState):
  # Match the flag or the specialty string from your RoutingAgent
  msg ="🚨 URGENZA: I tuoi sintomi suggeriscono un'emergenza. Chiama il 112 o recati al pronto soccorso piu' vicino."
  return {"messages": [("ai", msg)], "emergency_flag": True}


# DEFINE THE LOGIC (CONDITIONAL EDGES)

def route_decision(state: AgentState):
  # Match the flag or the specialty string from your RoutingAgent
  if state.get("specialty_required") == "EMERGENCY_SERVICES" or state.get("emergency_flag"):
    return "emergency"
  return END # We end the graph here and let main.py handle the search


# ASSEMBLE THE GRAPH

# 1. Initialize Graph with our STate schema
workflow = StateGraph(AgentState)

# 2. Add Notes
workflow.add_node("intake", intake_node)
workflow.add_node("router", routing_node)
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
    "search": END # Logic moves back to main.py
  }
)

# 5. Finish
workflow.add_edge("emergency", END)

# Compile the graph
app = workflow.compile()

