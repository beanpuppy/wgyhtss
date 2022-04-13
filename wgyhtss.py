import requests
import os
import logging

from fastai.learner import load_learner
from multiprocessing import Process, Queue
from datetime import datetime

from util import (
    AUDIO_DIR,
    SEGMENT_DIR,
    SPEC_DIR,
    record_wav,
    create_spectrogram,
    ignore_stderr
)

SERVER_IP = "192.168.1.169:5000"
SERVER_KEY = os.environ.get("WGYHTSS_KEY", None)

logger = logging.getLogger("wgyhtss")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

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
        pred, pred_idx, probs = learn_inf.predict(path)

        logger.debug(f"prediction: {pred} [{probs[pred_idx]:.2%}]")

        if pred == "scream" and probs[pred_idx] > 0.85 and SERVER_KEY:
            requests.get(r"http://{SERVER_IP}/dos?key={SERVER_KEY}", timeout=60)

            # Clear queue, sending a req will cause a lot to be in the backlog
            # we want to deal with it fresh
            while not queue.empty():
                queue.get()

if __name__ == "__main__":
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    SEGMENT_DIR.mkdir(parents=True, exist_ok=True)
    SPEC_DIR.mkdir(parents=True, exist_ok=True)

    learn_inf = load_learner("export.pkl")

    queue = Queue()

    record_process = Process(target=record_audio, args=(queue,))
    predict_process = Process(target=predict_queue, args=(queue, learn_inf))

    record_process.start()
    predict_process.start()

    record_process.join()
    predict_process.join()
