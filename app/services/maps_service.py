import googlemaps
import os
from dotenv import load_dotenv

load_dotenv()

class MapsService:
  def __init__(self):
    self.gmaps = googlemaps.Client(key=os.getenv("GOOGLE_MAPS_API_KEY"))


  def find_nearby_doctors(self, lat: float, lng: float, specialty: str, radius_meters: int = 10000):
    """
    Search for doctors/clinics nearby using a keyword (specialty).
    """

    # Mapping English specialties to Italian search terms
    query_map = {
      "General Practice": "Medico di base",
      "Pediatrics": "Pediatra",
      "Dermatology": "Dermatologo",
      "Dentist": "Dentista",
      "Emergency": "Pronto Soccorso",
      "Orthopedics": "Ortopedico",
      "Gynecology": "Ginecologo",
      "Psychiatry": "Psichiatra"
    }

    base_term = query_map.get(specialty, "Medico")

    # Use a "broad net" query string
    search_query = f"{base_term}"
    print(f"DEBUG: Searching for: {search_query} near {lat}, {lng}")

    # We use 'doctor' as the type and the specialty (e.g., 'Cardiologist') as the keyword
    places_result = self.gmaps.places(
      query=search_query,
      location=(lat, lng),
      radius=radius_meters,
      language='it'
    )

    results = places_result.get('results', [])
    print(f"DEBUG: Google found {len(results)} raw results.")

    doctors = []
    for place in results:
      dest_coords = place.get("geometry", {}).get("location")
      travel = self.get_travel_info((lat, lng), dest_coords)

      # Get real-time distance/time for each doctor found
      travel = self.get_travel_info((lat, lng), dest_coords)

      doctors.append({
        "id": place.get("place_id"),
        "place_id": place.get("place_id"),
        "name": place.get("name"),
        "address": place.get("formatted_address") or place.get("vicinity"), # Formatted address is for text search
        "rating": place.get("rating", 0),
        "category": specialty,
        "distance": travel["distance"],
        "isRegistered": False,
        "location": dest_coords
      })

    return doctors
  
  def get_travel_info(self, origin: tuple, destination_coords: dict):
    if not destination_coords:
      return {"distance": "N/D", "duration": "N/D"}
    """
    Calculate the distance and time between the patient and a doctor.
    """
    try:
      matrix = self.gmaps.distance_matrix(
        origins=origin,
        destinations=destination_coords,
        mode="driving",
        language="it"
      )

      element = matrix['rows'][0]['elements'][0]
      if element.get('status') == 'OK':
        return {
          "distance": element['distance']['text'],
          "duration": element['duration']['text']
        }
    except Exception:
      pass

    return {"distance": "Calcolo...", "duration": "N/A"}