from app.services.maps_service import MapsService
from app.services.tts_service import TTSService

# Initialize once, share everywhere
maps_service = MapsService()
tts = TTSService()