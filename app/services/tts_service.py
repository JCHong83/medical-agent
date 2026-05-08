import os
import re
from google.cloud import texttospeech
from base64 import b64encode

class TTSService:
    def __init__(self):
        # Ensure your GOOGLE_APPLICATION_CREDENTIALS env var is set
        self.client = texttospeech.TextToSpeechClient()
        self.voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Neural2-F", # High-quality female Italian voice
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            pitch=0.0,
            speaking_rate=1.05 # Slightly faster for a more natural conversational flow
        )

    def speak(self, text: str) -> str:
        try:
            # 1. Validation: Ensure text is a string and not empty
            if not text or not isinstance(text, str):
                text = "I am sorry, there was an error loading the message."

            # 2. Safety Gate: Clean text of any accidental Markdown or special chars
            # This prevents the TTS from literally reading things like "**" or "###"
            clean_text = re.sub(r'[*#_~-]', '', text)
            
            # 3. Byte Limit Protection: Google TTS has a 5000 character limit per request.
            # We truncate to 4000 to be safe and avoid the 400 error.
            if len(clean_text) > 4000:
                print(f"⚠️ TTS Warning: Text too long ({len(clean_text)} chars). Truncating.")
                clean_text = clean_text[:3997] + "..."

            synthesis_input = texttospeech.SynthesisInput(text=clean_text)
            
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=self.voice,
                audio_config=self.audio_config
            )

            # Convert binary audio to base64 string for easy transport
            return b64encode(response.audio_content).decode("utf-8")
            
        except Exception as e:
            print(f"❌ TTS Service Error: {str(e)}")
            # Return an empty string so the frontend doesn't crash if audio fails
            return ""