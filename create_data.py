import glob

from pathlib import Path
from multiprocessing import Process, Queue
from datetime import datetime

from util import (
    AUDIO_DIR,
    SEGMENT_DIR,
    SPEC_DIR,
    record_wav,
    create_segments,
    create_spectrogram
)

RECORD_SECS = 60

def record_audio(queue):
    while True:
        filename = datetime.now().strftime("%Y%m%d%H%M%S") + ".wav"
        record_wav(filename, RECORD_SECS)
        queue.put(filename)

def segment_queue(queue):
    while True:
        audio_path = queue.get(block=True)
        paths = create_segments(audio_path)

        for path in paths:
            create_spectrogram(path)

def add_existing():
    # Create audio segments
    for file in glob.glob(str(AUDIO_DIR) + '/*.wav'):
        path = Path(file)
        create_segments(path)

    # Create spectrograms
    for file in glob.glob(str(SEGMENT_DIR) + '/*.wav'):
        path = Path(file)
        create_spectrogram(path)

if __name__ == '__main__':
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    SEGMENT_DIR.mkdir(parents=True, exist_ok=True)
    SPEC_DIR.mkdir(parents=True, exist_ok=True)

    # Also make training data directories
    Path('train/scream').mkdir(parents=True, exist_ok=True)
    Path('train/notscream').mkdir(parents=True, exist_ok=True)

    # add_existing()

    queue = Queue()

    record_process = Process(target=record_audio, args=(queue,))
    segment_process = Process(target=segment_queue, args=(queue,))

    record_process.start()
    segment_process.start()

    record_process.join()
    segment_process.join()
