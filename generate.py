import os
import pickle

import numpy as np
import soundfile as sf
from speechgenerator import SoundGenerator

from unet import UNET


# constants
HOP_LENGTH = 256
SAMPLE_RATE = 22050


SAVED_MODEL_DIR = '/content/drive/MyDrive/Colab Notebooks/speech denoiser/training/best_weight.hdf5'
MIN_MAX_VALUES_PATH = 'min_max_values.pkl'
SPECTROGRAMS_PATH = '/content/drive/MyDrive/Colab Notebooks/speech denoiser/x_train_noised_speech'
SAVE_DIR_ORIGINAL = "samples/original/"
SAVE_DIR_GENERATED = "samples/generated/"


def load_fsdd(spectrograms_path):
    arr = []
    file_paths = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path) # (n_bins, n_frames, 1)
            arr.append(spectrogram)
            file_paths.append(file_path)
    # Convert the list to a NumPy array if required
    arr = np.array(arr)

    # reshaping to make it suitable for training
    arr = arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2], 1 )
    return arr, file_paths


def select_spectrograms(spectrograms,
                        file_paths,
                        min_max_values,
                        num_spectrograms=2):
    sampled_indexes = np.random.choice(range(len(spectrograms)), num_spectrograms)
    sampled_spectrogrmas = spectrograms[sampled_indexes]
    file_paths = [file_paths[index] for index in sampled_indexes]
    sampled_min_max_values = [min_max_values[file_path] for file_path in
                           file_paths]
    print(file_paths)
    print(sampled_min_max_values)
    return sampled_spectrogrmas, sampled_min_max_values


def save_signals(signals, save_dir, sample_rate=22050):
    for i, signal in enumerate(signals):
        save_path = os.path.join(save_dir, str(i) + ".wav")
        sf.write(save_path, signal, sample_rate)


if __name__ == "__main__":
    # initialise sound generator
    model = UNET(
        input_shape=(256, 256, 1),
        conv_filters=(64, 128, 256, 512),
        conv_kernels=(3, 3),
    )
    model.load_weights(SAVED_MODEL_DIR)

    sound_generator = SoundGenerator(model, HOP_LENGTH)

    # load spectrograms + min max values
    with open(MIN_MAX_VALUES_PATH, "rb") as f:
        min_max_values = pickle.load(f)

    specs, file_paths = load_fsdd(SPECTROGRAMS_PATH)

    # sample spectrograms + min max values
    sampled_specs, sampled_min_max_values = select_spectrograms(specs,
                                                                file_paths,
                                                                min_max_values,
                                                                5)

    # generate audio for sampled spectrograms
    signals = sound_generator.generate(sampled_specs,
                                          sampled_min_max_values)

    # convert spectrogram samples to audio
    original_signals = sound_generator.convert_spectrograms_to_audio(
        sampled_specs, sampled_min_max_values)

    # save audio signals
    save_signals(signals, SAVE_DIR_GENERATED)
    save_signals(original_signals, SAVE_DIR_ORIGINAL)