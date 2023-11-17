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

import numpy as np
import os
import random
import pickle
import librosa


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
    


class Mixer:
    """Mixer is responsible for mixing two audios"""
    def __init__(self, overlap_ratio) -> None:
        self.overlap_ratio = overlap_ratio

    def mix_audio(self, clean_audio, noise_audio):
        mixed_audio = clean_audio + noise_audio[:len(clean_audio)] * self.overlap_ratio
        return mixed_audio
    


class LogSpectrogramExtractor:
    """Responsible for extraing the log spectrogram (in DB) of the 
    time series audio"""
    def __init__(self, frame_size, hop_length) -> None:
        self.frame_size = frame_size
        self.hop_length = hop_length

    def extract(self, signal):
        stft = librosa.stft(signal,
                                n_fft=self.frame_size,
                                hop_length=self.hop_length)
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        return log_spectrogram
    

class MinMaxNormalizer:
    """MinMaxNormaliser applies min max normalisation to an spectrogram(array)"""
    def __init__(self) -> None:
        pass

    def normalize(self, spectrogram, min_val, max_val):
        normalized_spectrogram = (spectrogram - min_val) / (max_val - min_val)
        return normalized_spectrogram

    def denormalize(self, spectrogram, original_min, original_max):
        denormalized_spectrogram = spectrogram * (original_max - original_min) + original_min
        return denormalized_spectrogram


class Saver:
    """saver is responsible to save features, min max values, and pickle file."""
    def __init__(self, clean_feature_folder, noise_feature_folder) -> None:
        self.clean_feature_foler = clean_feature_folder
        self.noise_feature_foler = noise_feature_folder

    def save_normalized_spectrogram(self, spectrogram, filename, clean):
        if clean:
            save_path = os.path.join(self.clean_feature_foler, f"{filename}.npy")
            np.save(save_path, spectrogram)
            return save_path
        else:
            save_path = os.path.join(self.noise_feature_foler, f"{filename}.npy")
            np.save(save_path, spectrogram)
            return save_path
        
    
    def save_min_max_value(self, dir, data):
        save_path = os.path.join(dir, "min_max_values.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)


class PreprocessingPipeline:
    """PreprocessingPipeline processes audio files in a directory, applying
    the following steps to each file:
        1- load a file
        2- trim the signal
        3- mix signal with noise for training
        3- extracting log spectrogram from signal
        4- normalise spectrogram
        5- save the normalised spectrogram

    Storing the min max values for all the log spectrograms.
    """
    def __init__(self):
        self.loader = None
        self.trimmer = None
        self.mixer = None
        self.extractor = None
        self.normaliser = None
        self.saver = None
        self.min_max_values = {}

    def process(self, clean_audio_dir, noise_audio_dir, min_max_value_dir):
        for filename in os.listdir(clean_audio_dir):
            if filename.endswith(".wav"):
                clean_audio_path = os.path.join(clean_audio_dir, filename)
                noise_audio_path = random.choice(os.listdir(noise_audio_dir))
                noise_audio_path = os.path.join(noise_audio_dir, noise_audio_path)
                self._process_file(filename, clean_audio_path, noise_audio_path)

        self.saver.save_min_max_value(min_max_value_dir,self.min_max_values)

    
    def _process_file(self,filename,  clean_audio_path, noise_audio_path):
        clean_audio = self.loader.load(clean_audio_path)
        trimmed_clean_audio = self.trimmer.trim_audio(clean_audio)

        noise_audio = self.loader.load(noise_audio_path)
        trimmed_noise_audio = self.trimmer.trim_audio(noise_audio)

        mixed_audio = self.mixer.mix_audio(trimmed_clean_audio, trimmed_noise_audio)

        log_spec_noisy = self.extractor.extract(mixed_audio)
        log_spec_clean = self.extractor.extract(trimmed_clean_audio)

        normalized_spec_noisy = self.normaliser.normalize(log_spec_noisy, np.min(log_spec_noisy), np.max(log_spec_noisy))
        normalized_spec_clean = self.normaliser.normalize(log_spec_clean, np.min(log_spec_clean), np.max(log_spec_clean))

        save_path_noisy = self.saver.save_normalized_spectrogram(normalized_spec_noisy, f'{filename}_spec', False)
        save_path_clean = self.saver.save_normalized_spectrogram(normalized_spec_clean, f'{filename}_spec', True)

        self._store_min_max_value(save_path_noisy, log_spec_noisy.min(), log_spec_noisy.max())
        print(f"Processed Noisy file: {save_path_noisy}, Processed Clean file: {save_path_clean}")

    def _store_min_max_value(self, save_path, min_val, max_val):
        self.min_max_values[save_path] = {
            "min": min_val,
            "max": max_val
        }



    
if __name__ == "__main__":
    SAMPLE_RATE = 22050
    MONO = True
    HOP_LENGTH = 256
    TOP_DB = 20
    FRAME_SIZE = 512
    OVERLAP_RATIO=0.3

    # Defining the paths to the clean audio and noise audio folders
    clean_audio_dir = 'D:/Speech data/MS-SNSD/clean_train'
    noise_audio_dir = 'D:/Speech data/MS-SNSD/noise_train'

    # folder for storing spectogram value of noised and clean audio speech
    X_train_spec_dir = 'x_train_noised_speech'
    Y_train_spec_dir = 'y_train_clean_audio'

    # folder for storing min max values dictionary
    min_max_value_dir = 'min_max_value_save'

    # Ensure the output folders exist
    if not os.path.exists(X_train_spec_dir):
        os.makedirs(X_train_spec_dir)
    if not os.path.exists(Y_train_spec_dir):
        os.makedirs(Y_train_spec_dir)
    if not os.path.exists(min_max_value_dir):
        os.makedirs(min_max_value_dir)

    # instantiate all objects
    loader = Loader(sample_rate=SAMPLE_RATE, 
                    mono=MONO)
    trimmer = Trimmer(hop_length=HOP_LENGTH, 
                      top_db=TOP_DB)
    mixer = Mixer(overlap_ratio=OVERLAP_RATIO)
    log_spectrogram_extractor = LogSpectrogramExtractor(frame_size=FRAME_SIZE, 
                                                        hop_length=HOP_LENGTH)
    min_max_normalizer = MinMaxNormalizer()
    saver = Saver(clean_feature_folder=Y_train_spec_dir,
                  noise_feature_folder=X_train_spec_dir,)

    preprocessing_pipeline = PreprocessingPipeline()
    preprocessing_pipeline.loader = loader
    preprocessing_pipeline.trimmer = trimmer
    preprocessing_pipeline.mixer = mixer
    preprocessing_pipeline.extractor = log_spectrogram_extractor
    preprocessing_pipeline.normaliser = min_max_normalizer
    preprocessing_pipeline.saver = saver

    preprocessing_pipeline.process(clean_audio_dir=clean_audio_dir,
                                   noise_audio_dir=noise_audio_dir,
                                   min_max_value_dir=min_max_value_dir)



    

    
