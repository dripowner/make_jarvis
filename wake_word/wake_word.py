import pvporcupine
from dotenv import load_dotenv, dotenv_values
import os
from pvrecorder import PvRecorder
import wave
import struct
from datetime import datetime
from utils import wav_to_mp3
import whisper


load_dotenv() 

porcupine = pvporcupine.create(
	access_key=os.getenv("ACCESS_KEY"),
	keyword_paths=[os.getenv("KEYWORD_FILE_PATH")],
	model_path=os.getenv("MODEL_FILE_PATH"),
	sensitivities=[1.0]
)
recorder = PvRecorder(
	frame_length=porcupine.frame_length,
	device_index=0)

keywords = list()
for x in os.getenv("KEYWORD_FILE_PATH"):
    keyword_phrase_part = os.path.basename(x).replace('.ppn', '').split('_')
    if len(keyword_phrase_part) > 6:
        keywords.append(' '.join(keyword_phrase_part[0:-6]))
    else:
        keywords.append(keyword_phrase_part[0])

output_path = "out.wav"
model = whisper.load_model("base")

if __name__ == "__main__":
	recorder.start()
	print('Listening ... (press Ctrl+C to exit)')
	wav_file = None
	if output_path is not None:
		wav_file = wave.open(output_path, "w")
		wav_file.setnchannels(1)
		wav_file.setsampwidth(2)
		wav_file.setframerate(16000)
	try:
		while True:
			pcm = recorder.read()
			result = porcupine.process(pcm)

			if wav_file is not None:
				wav_file.writeframes(struct.pack("h" * len(pcm), *pcm))

			if result >= 0:
				print('[%s] Detected %s' % (str(datetime.now()), keywords[result]))
				wav_to_mp3(output_path, output_path[:-3]+"mp3")
				result = model.transcribe(output_path[:-3]+"mp3", language="ru", task="transcribe")
				print(result["text"])
				break
	except KeyboardInterrupt:
		print('Stopping ...')
	finally:
		recorder.delete()
		porcupine.delete()
		if wav_file is not None:
			wav_file.close()
	