import pvporcupine
from dotenv import load_dotenv
import os
from pvrecorder import PvRecorder
import wave
from datetime import datetime
from pydub import AudioSegment
import glob
import time
import struct

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from stt.STT import STT
from llm.LLM import LLM
from tts.TTS import TTS

class Recorder():
    def __init__(self,
                 access_key,
                 keyword_paths,
                 model_path,
                 save_count=10,
                 chunk_size=20,
                 ) -> None:
        load_dotenv()
        self.save_count = save_count # max records
        self.records_dir = "./records"
        self.wav_record_dir = os.path.join(self.records_dir, "wav")
        if not os.path.exists(self.wav_record_dir):
            os.makedirs(self.wav_record_dir)
        self.mp3_record_dir = os.path.join(self.records_dir, "mp3")
        if not os.path.exists(self.mp3_record_dir):
            os.makedirs(self.mp3_record_dir)
        self.porcupine = pvporcupine.create(
            access_key=access_key,
            keyword_paths=[keyword_paths],
            model_path=model_path,
            sensitivities=[1.0]
        )
        self.recorder = PvRecorder(
            frame_length=self.porcupine.frame_length,
            device_index=0)
        self.keywords = list()
        for x in keyword_paths:
            keyword_phrase_part = os.path.basename(x).replace('.ppn', '').split('_')
            if len(keyword_phrase_part) > 6:
                self.keywords.append(' '.join(keyword_phrase_part[0:-6]))
            else:
                self.keywords.append(keyword_phrase_part[0])
        
        self.stt = STT()
        self.chunk_size = chunk_size
        self.llm = LLM()
        self.tts = TTS(os.getenv("ELEVEN_LABS"))

    def wav2mp3(self, wav_file_path, mp3_file_path):
        sound = AudioSegment.from_wav(wav_file_path)
        sound.export(mp3_file_path, format="mp3")
        return None
    
    def delete_oldest_file(self, path):
        if len(os.listdir(path)) > self.save_count:
            files = glob.glob(path + '/*')
            oldest_file = min(files, key=os.path.getctime)
            os.remove(oldest_file)
        return None
    
    def text_response(self, request):
        return self.llm.answer(request)
    
    def record_one(self):
        try:
            self.recorder.start()
            start_time = time.time()
            wav_file = None
            wav_file_name = str(datetime.now()).replace(" ", "_").replace(".", "_").replace(":", "_")
            output_path = os.path.join(self.wav_record_dir, wav_file_name + ".wav")
            if output_path is not None:
                wav_file = wave.open(output_path, "w")
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
        
            while True:
                pcm = self.recorder.read()
                result = self.porcupine.process(pcm)

                if wav_file is not None:
                    wav_file.writeframes(struct.pack("h" * len(pcm), *pcm))
                    
                if time.time() - start_time > self.chunk_size:
                    break

                if result >= 0:
                    print('[%s] Detected %s' % (str(datetime.now()), self.keywords[result]))
                    self.wav2mp3(output_path, os.path.join(self.mp3_record_dir, wav_file_name + ".mp3"))
                    self.delete_oldest_file(self.wav_record_dir)
                    self.delete_oldest_file(self.mp3_record_dir)
                    result = self.stt.transcribe(os.path.join(self.mp3_record_dir, wav_file_name + ".mp3"))
                    print(f"Request: {result}")
                    answer = self.text_response(result)
                    self.tts.play(answer)
                    print(f"Answer: {answer}")
                    break
        except KeyboardInterrupt:
            self.recorder.delete()
            self.porcupine.delete()
            if wav_file is not None:
                wav_file.close()
            
        

    def record(self):
        print('Listening ... (press Ctrl+C to exit)')
        try:
            while True:
                self.record_one()
        except KeyboardInterrupt:
            print('Stopping ...')