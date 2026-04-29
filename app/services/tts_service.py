import os
from google.cloud import texttospeech
from base64 import b64encode

class TTSService:
  def __init__(self):
    # Ensure your GOOGLE_APPLICATION_CREDENTIALS anv var is set
    self.client = texttospeech.TextToSpeechClient()
    self.voice = texttospeech.VoiceSelectionParams(
      language_code="it-IT",
      name="it-IT-Neural2-A", # High-quality female Italian voice
      ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )
    self.audio_config = texttospeech.AudioConfig(
      audio_encoding=texttospeech.AudioEncoding.MP3
    )

  def speak(self, text: str) -> str:
    synthesis_input = texttospeech.SynthesisInput(text=text)
    response = self.client.synthesize_speech(
      input=synthesis_input,
      voice=self.voice,
      audio_config=self.audio_config
    )
    # Convert binary audio to base64 string for easy transport
    return b64encode(response.audio_content).decode("utf-8")