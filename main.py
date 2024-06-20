from wake_word.Recorder import Recorder
from dotenv import load_dotenv
import os

if __name__ == "__main__":
    load_dotenv(override=True)
    recorder = Recorder(os.getenv("ACCESS_KEY"),
                        os.getenv("KEYWORD_FILE_PATH"),
                        os.getenv("MODEL_FILE_PATH"))
    recorder.record()