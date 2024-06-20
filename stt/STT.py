import whisper

class STT():
    def __init__(self, whisper_model="small") -> None:
        self.model = whisper.load_model(whisper_model)
    
    def transcribe(self, mp3_file_path):
        result = self.model.transcribe(mp3_file_path, language="ru", task="transcribe")
        return result["text"]
