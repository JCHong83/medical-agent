from app.schemas.state import AgentState
from app.services.maps_service import MapsService

class SerachNode:
  def __init__(self):
    self.maps_service = MapsService()

  async def call_node(self, state: AgentState):
    """
    LangGraph Node: Takes the required specialty and location to find real doctors.
    """
    # Extract requirements from state
    specialty = state.get("specialty_required")
    location = state.get("patient_location") # Handed over from React Native

    # Safety Check: If it's an emergency, we might want to search for 'ER' specifically
    if state.get("emergency_flag") or specialty == "Emergency Room":
      search_query = "Hospital Emergency Room"
    else:
      search_query = specialty
    
    # Handle missing location
    if not location or not location.get("lat") or not location.get("lng"):
      print("[SearchNode] Warning: No patient location found. Cannot perform search.")
      return {
        "recommended_doctors": [],
        "messages": [("assistant", "I've identified that you need a specialist, but I couldn't access your location to find clinics nearby.")]
      }
    
    print(f"[SearchNode] Searching for {search_query} near {location['lat']}, {location['lng']}")

    # Execute the Search
    try:
      doctors = await self.gmaps_service.find_nearby_doctors(
        lat=location["lat"],
        lng=location["lng"],
        specialty=search_query
      )

      # Prepare a helpful summary message
      if doctors:
        summary = f"I've found {len(doctors)} {search_query} locations near you."
      else:
        summary = f"I identified you need a {search_query}, but I couldn't find any specific clinics within 10km of your current location."

      return {
        "recommended_doctors": doctors,
        "messages": [("assistant", summary)]
      }
    
    except Exception as e:
      print(f"[SearchNode] Error during search: {e}")
      return {"recommended_doctors": []}