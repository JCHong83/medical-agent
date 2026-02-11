import asyncio
from langchain_core.messages import HumanMessage
from app.agents.graph import app # Import the compiled graph

async def run_test(user_query: str):
  print(f"\n--- Testing with query: '{user_query}' ---")

  # Initial state matching our AgentState schema
  initial_state = {
    "messages": [HumanMessage(content=user_query)],
    "symptoms": [],
    "duration": "",
    "severity": "",
    "emergency_flag": False,
    "specialty_required": "",
    "patient_location": {"lat": 45.4642, "lng": 9.1900}, # Default to Milan for testing
    "recommended_doctors": []
  }

  # We use astream to see the updates in real-time
  async for event in app.astream(initial_state, stream_mode="updates"):
    for node_name, state_update in event.items():
      print(f"\n📍 Node: {node_name}")
      for key, value in state_update.items():
        print(f"  └─ Update {key}: {value}")

async def main():
  # Test Case 1: Standard Symptom
  await run_test("I have a red rash on my arm that is very itchy.")

  # Test Case 2: Emergency Symptom
  await run_test("I am having sudden sharp chest pain and trouble breathing.")

if __name__ == "__main__":
  asyncio.run(main())