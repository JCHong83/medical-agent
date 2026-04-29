
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


# --- Define Conditional Logic (Edge Functions) ---

def loop_decision(state: AgentState):
  if state.get("is_gathering_complete"):
    return "proceed_to_routing"
  return "ask_user_more"

def triage_decision(state: AgentState):
  if state.get("specialty_required") == "EMERGENCY_SERVICES" or state.get("emergency_flag"):
    return "emergency"
  return "search"


# ASSEMBLE THE GRAPH

# Initialize Graph with our STate schema
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("intake", intake_node)
workflow.add_node("router", routing_node)
workflow.add_node("emergency", emergency_node)

# Add Edges (The Flow)
workflow.add_edge(START, "intake")

# INTAKE -> Loop back to User or Move to Router
workflow.add_conditional_edges(
  "router",
  loop_decision,
  {
    "proceed_to_routing": "router",
    "ask_user_more": END
  }
)

# ROUTER -> Emergency path or Standard search path
workflow.add_conditional_edges(
  "router",
  triage_decision,
  {
    "emergency": "emergency",
    "search": END
  }
)

# 5. Finish
workflow.add_edge("emergency", END)

# Compile the graph
app = workflow.compile()

