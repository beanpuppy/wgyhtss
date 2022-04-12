from pathlib import Path
from scipy.io import wavfile
from pydub import AudioSegment

import matplotlib.pyplot as plt
import math
import pyaudio
import wave

SECS_PER_SPLIT = 15

CHUNK_SIZE = 1024
RATE = 44100
CHANNELS = 2

FORMAT = pyaudio.paInt16

AUDIO_DIR = Path('./data/audio/full')
SEGMENT_DIR = Path('./data/audio/segement')
SPEC_DIR = Path('./data/spectrograms')

def single_split(audio, from_sec, to_sec, file):
    t1 = from_sec * 1000
    t2 = to_sec * 1000
    split_audio = audio[t1:t2]
    split_audio.export(file, format='wav')

def create_segments(path):
    audio = AudioSegment.from_wav(path)
    duration = math.ceil(audio.duration_seconds)  # type: ignore
    paths = []

    for i in range(0, duration, SECS_PER_SPLIT):
        file = SEGMENT_DIR / f'{path.stem}_seg_{str(i)}.wav'
        single_split(audio, i, i + SECS_PER_SPLIT, file)
        paths.append(file)

    return paths

def create_spectrogram(wav_file):
    path = SPEC_DIR / f'{wav_file.stem}.png'
    rate, audio = wavfile.read(wav_file)
    audio = audio[:,0]  # select left channel only
    fig, ax = plt.subplots(1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.specgram(x=audio, Fs=rate, noverlap=384, NFFT=512)
    ax.axis('off')
    fig.savefig(path, dpi=300)
    return path

def record_wav(filename, seconds):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    frames_per_buffer=CHUNK_SIZE,
                    input=True)

    frames = []

    for _ in range(0, int(RATE / CHUNK_SIZE * seconds)):
        data = stream.read(CHUNK_SIZE)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(str(AUDIO_DIR / filename), 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
