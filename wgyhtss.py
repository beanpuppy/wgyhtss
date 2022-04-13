import requests
import os

from fastai.learner import load_learner
from multiprocessing import Process, Queue
from datetime import datetime

from util import (
    AUDIO_DIR,
    record_wav,
    create_spectrogram,
    ignore_stderr
)

SERVER_IP = "192.168.1.169:5000"
SERVER_KEY = os.environ["WGYHTSS_KEY"]

def record_audio(queue):
    while True:
        filename = datetime.now().strftime("%Y%m%d%H%M%S") + ".wav"

        # Ignore stderr so it's easier to see preditions
        # They aren't actually errors
        with ignore_stderr():
            record_wav(filename, 15)

        queue.put(filename)

def predict_queue(queue, learn_inf):
    while True:
        audio_path = queue.get(block=True)
        path = create_spectrogram(AUDIO_DIR / audio_path)
        prediction = learn_inf.predict(path)

        print(prediction)

        if prediction[0] == "scream":
            requests.get(r"http://{SERVER_IP}/dos?key={SERVER_KEY}", timeout=60)

if __name__ == "__main__":
    learn_inf = load_learner("export.pkl")

    queue = Queue()

    record_process = Process(target=record_audio, args=(queue,))
    predict_process = Process(target=predict_queue, args=(queue, learn_inf))

    record_process.start()
    predict_process.start()

    record_process.join()
    predict_process.join()
