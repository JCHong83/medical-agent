import googlemaps
import os
from dotenv import load_dotenv
from typing import List, Dict, Any

load_dotenv()

class MapsService:
  def __init__(self):
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
      print("⚠️ WARNING: GOOGLE_MAPS_API_KEY not found in environment.")
    self.gmaps = googlemaps.Client(key=api_key) if api_key else None

  async def find_nearby_doctors(self, lat: float, lng: float, specialty: str, radius_meters: int = 10000) -> List[Dict[str, Any]]:
    """
    Search for clinics nearby. Increased deafult radius to 10km for better results.
    """
    if not self.gmaps:
      return []
    
    # Broaden the keyword to capture both the specialty and the facility
    search_query = f"{specialty} clinic hospistal"

    try:
      places_result = self.gmaps.places_nearby(
        location=(lat, lng),
        radius=radius_meters,
        keyword=search_query,
        type='health' # 'health' or 'doctor' are standard Google types
      )

      doctors = []
      results = places_result.get('results', [])[:5]

      for place in results:
        dest_coords = place.get("geometry", {}).get("location")

        # Fetch travel info immediately to provide a complete 'distance'
        travel = self.get_travel_info((lat, lng), dest_coords)

        doctors.append({
          "name": place.get("name"),
          "address": place.get("vicinity"),
          "rating": place.get("rating", 0),
          "distance": travel["distance"],
          "duration": travel["duration"],
          "location": dest_coords,
          "place_id": place.get("place_id"),
        })

      return doctors
    except Exception as e:
      print(f"Error fetching Google Places: {e}")
      return []
  
  def get_travel_info(self, origin: dict, destination: dict) -> Dict[str, str]:
    """
    Calculate distance/time. Using a tuple for origin (lat, lng).
    """
    if not self.gmaps or not destination:
      return {"distance": "N/A", "duration": "N/A"}
    
    try:
      matrix = self.gmaps.distance_matrix(
        origins=[origin],
        destinations=[(destination['lat'], destination['lng'])],
        mode="driving"
      )

      element = matrix['rows'][0]['elements'][0]

      if element.get('status') == 'OK':
        return {
          "distance": element['distance']['text'],
          "duration": element['duration']['text']
        }
    except Exception:
      pass

    return {"distance": "Nearby", "duration": "Calculating..."}
    