import googlemaps
import os
from dotenv import load_dotenv

load_dotenv()

class MapsService:
  def __init__(self):
    self.gmaps = googlemaps.Client(key=os.getenv("GOOGLE_MAPS_API_KEY"))

  def find_nearby_doctors(self, lat: float, lng: float, specialty: str, radius_meters: int = 5000):
    """
    Search for doctors/clinics nearby using a keyword (specialty).
    """
    # We use 'doctor' as the type and the specialty (e.g., 'Cardiologist') as the keyword
    places_result = self.gmaps.places_nearby(
      location=(lat, lng),
      radius=radius_meters,
      keyword=specialty,
      type='doctor'
    )

    doctors = []
    for place in places_result.get('results', [])[:5]: # Get top 5 results
      doctors.append({
        "name": place.get("name"),
        "address": place.get("vicinity"),
        "place_id": place.get("place_id"),
        "rating": place.get("rating"),
        "location": place.get("geometry", {}).get("location")
      })
    return doctors
  
  def get_travel_info(self, origin: dict, destination_coords: dict):
    """
    Calculate the distance and time between the patient and a doctor.
    """
    matrix = self.gmaps.distance_matrix(
      origins=origin,
      destinations=destination_coords,
      mode="driving"
    )

    try:
      element = matrix['rows'][0]['elements'][0]
      return {
        "distance": element['distance']['text'],
        "duration": element['duration']['text']
      }
    except (KeyError, IndexError):
      return {"distance": "Unknown", "duration": "Unknown"}