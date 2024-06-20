import os
from pydub import AudioSegment

# ffmpeg needed
def wav_to_mp3(wav_file_path, mp3_file_path):
    sound = AudioSegment.from_wav(wav_file_path)
    sound.export(mp3_file_path, format="mp3")
    return