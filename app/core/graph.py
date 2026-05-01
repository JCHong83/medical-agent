from langgraph.graph import StateGraph, START, END
from app.core.state import AgentState
from app.agents.intake_agent import IntakeAgent

# Initialize our component
intake = IntakeAgent()

# --- NODE FUNCTIONS ---

async def intake_node(state: AgentState):
    # This node now handles both extraction and specialty identification
    return await intake.call_node(state)

async def emergency_node(state: AgentState):
    msg = "🚨 URGENZA: I tuoi sintomi suggeriscono un'emergenza. Chiama il 112 o recati al pronto soccorso più vicino."
    return {"messages": [("ai", msg)], "emergency_flag": True, "is_gathering_complete": True}

# --- CONDITIONAL LOGIC (EDGE FUNCTIONS) ---

def after_intake_decision(state: AgentState):
    """
    Determines if we need more info, if it's an emergency, or if we search.
    """
    if state.get("emergency_flag") is True:
        return "emergency"
    
    if state.get("is_gathering_complete") is True:
        return "finish_and_search"
    
    # If not finished and no emergency, go to END to wait for next user voice input
    return "wait_for_user"

# --- ASSEMBLE THE GRAPH ---

workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("intake", intake_node)
workflow.add_node("emergency", emergency_node)

# START -> INTAKE
workflow.add_edge(START, "intake")

# INTAKE -> Decide where to go
workflow.add_conditional_edges(
    "intake",
    after_intake_decision,
    {
        "emergency": "emergency",
        "finish_and_search": END, # main.py will see is_gathering_complete=True and trigger search
        "wait_for_user": END      # main.py will see is_gathering_complete=False and play audio
    }
)

workflow.add_edge("emergency", END)

# Compile the graph
app = workflow.compile()