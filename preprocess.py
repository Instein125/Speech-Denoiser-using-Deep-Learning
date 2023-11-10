"""
1. load clean and noise audio files

for each file in clean audio file perform the following steps:
2. trim clean and noise audio files
3. mix clean audio with random noise
4. generate log spectogram of noisy speech
5. mix max normalizarion of noisy speech
6. save log spectogram in X_train_spec folder
7. store orginal min max value of each log spectogram as dictionary {'save_path':{'min': ,'max': }}

finally save the pickle file of min max value
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import os
import soundfile as sf
import random
import pickle

from glob import glob

import librosa
import librosa.display
import IPython.display as ipd

class Loader:
    """Loader is responsible for loading an audio file."""
    def __init__(self, sample_rate, mono):
        self.sample_rate = sample_rate
        self.mono = mono

    def load(self, file_path):
        signal = librosa.load(file_path,
                              sr=self.sample_rate,
                              mono=self.mono)[0]
        return signal
    
class Trimmer:
    """Trimmer is responsible for triming the silence in audio"""
    def __init__(self, top_db = 20, hop_length = 256) -> None:
        self.top_db = top_db
        self.hop_length = hop_length

    def trim_audio(self, audio):
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=self.top_db, hop_length=self.hop_length)
        return audio_trimmed
    
if __name__ == "__main__":
    SAMPLE_RATE = 22050
    MONO = True
    HOP_LENGTH = 256
    TOP_DB = 20

    file_path = 'clean_audio/p234_001.wav'
    loader = Loader(sample_rate=SAMPLE_RATE, mono=MONO)
    trimmer = Trimmer(hop_length=HOP_LENGTH, top_db=TOP_DB)

    audio = loader.load(file_path=file_path)
    trim_audio = trimmer.trim_audio(audio)


    print(trim_audio.shape, audio.shape)
    

    
