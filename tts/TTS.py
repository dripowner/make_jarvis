from elevenlabs import play
from elevenlabs.client import ElevenLabs

class TTS():
    def __init__(self, api_key):
        self.client = ElevenLabs(api_key=api_key)
    
    def play(self, text):
        audio = self.client.generate(
            text=text,
            voice="Rachel",
            model='eleven_multilingual_v1'
        )
        play(audio)