import streamlit as st
import numpy as np
from scipy.io.wavfile import write
import librosa
import matplotlib.pyplot as plt
import pandas as pd
from unet import UNET
import soundfile as sf


# """
# 1. Select noised audio file
# 2. Load noised audio file
# 3. Extract spectrogram
# 4. Normalize and denormalize
# 4. Change the shape of spectrogram
# 5. Create a unet model with desired shape
# 6. Load model weights
# 7. Contruct the denoised audio
# 8. Plot the waveform, spectrogram
# """

def load_audio(y):
    return librosa.load(y)

def extract(signal, frame_size=512, hop_length=256):
        stft = librosa.stft(signal,
                                n_fft=frame_size,
                                hop_length=hop_length)
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        return log_spectrogram

def normalize(spectrogram, min_val, max_val):
    normalized_spectrogram = (spectrogram - min_val) / (max_val - min_val)
    return normalized_spectrogram

def denormalize(spectrogram, original_min, original_max):
    denormalized_spectrogram = spectrogram * (original_max - original_min) + original_min
    return denormalized_spectrogram

def load_numpy(spectrogram, final_shape):
    arr = []
    if spectrogram.shape[1] < final_shape[1]:
    # Pad the spectrogram to match the desired final shape
        pad_width = ((0, 0), (0, final_shape[1] - spectrogram.shape[1]))
        processed_spectrogram = np.pad(spectrogram, pad_width, mode='constant', constant_values=0)
    elif spectrogram.shape[1] > final_shape[1]:
     # Trim the spectrogram to match the desired final shape
        processed_spectrogram = spectrogram[:final_shape[0], :final_shape[1]]
    else:
        processed_spectrogram = spectrogram  # No change needed if the shape is already as desired
    
    # Append the padded spectrogram to the x_train list
    processed_spectrogram = processed_spectrogram[:final_shape[0], :]
    arr.append(processed_spectrogram)
    # Convert the list to a NumPy array if required
    arr = np.array(arr)
    # reshaping to make it suitable for training
    arr = arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2], 1 )
    # print(arr.shape)
    return arr

def create_model(final_shape):
    model = UNET(
        input_shape=(final_shape[0], final_shape[1], 1),
        conv_filters=(64, 128, 256, 512),
        conv_kernels=(3, 3),
    )
    model.load_weights('best_weight.hdf5')
    return model

def convert_spectrogram_to_audio( spectrogram, min_val, max_val):
        
        # reshape the log spectrogram
        spectrogram = spectrogram.reshape(spectrogram.shape[0], spectrogram.shape[1],)
        # apply denormalisation
        denorm_log_spec = denormalize(
            spectrogram, min_val, max_val)
        # log spectrogram -> spectrogram
        spec = librosa.db_to_amplitude(denorm_log_spec)
        # apply Griffin-Lim
        signal = librosa.istft(spec, hop_length=256)
        return signal, spectrogram
        return signal, spectrogram

def plot_waveform(y, title="Waveform"):
    # Generate time values for x-axis

    # Plot the waveform
    plt.figure(figsize=(10, 5))
    pd.Series(y).plot(
                  lw=1,
                  title=title,
                 )
    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    st.pyplot(plt)


def plot_spect(y, title):
    # Plot the spectogram
    fig, ax = plt.subplots(figsize=(10, 5))
    img = librosa.display.specshow(y,
                                x_axis='time',
                                y_axis='log',
                                ax=ax)
    ax.set_title(title, fontsize=20)
    fig.colorbar(img, ax=ax, format=f'%0.2f')
    st.pyplot(plt)

def resize_spectrogram(original_shape, processed_spectrogram):
    # Extract the original shape
    original_rows, original_cols = original_shape

    # Extract the processed shape
    processed_rows, processed_cols = processed_spectrogram.shape

    # Calculate the row and column differences
    row_diff = original_rows - processed_rows
    col_diff = original_cols - processed_cols

    # Pad or crop the processed spectrogram to match the original shape
    if row_diff > 0:
        # Pad along the rows
        pad_width = ((0, row_diff), (0, 0))
        processed_spectrogram = np.pad(processed_spectrogram, pad_width, mode='constant', constant_values=0)
    elif row_diff < 0:
        # Crop along the rows
        processed_spectrogram = processed_spectrogram[:original_rows, :]

    if col_diff > 0:
        # Pad along the columns
        pad_width = ((0, 0), (0, col_diff))
        processed_spectrogram = np.pad(processed_spectrogram, pad_width, mode='constant', constant_values=0)
    elif col_diff < 0:
        # Crop along the columns
        processed_spectrogram = processed_spectrogram[:, :original_cols]

    return processed_spectrogram
def resize_spectrogram(original_shape, processed_spectrogram):
    # Extract the original shape
    original_rows, original_cols = original_shape

    # Extract the processed shape
    processed_rows, processed_cols = processed_spectrogram.shape

    # Calculate the row and column differences
    row_diff = original_rows - processed_rows
    col_diff = original_cols - processed_cols

    # Pad or crop the processed spectrogram to match the original shape
    if row_diff > 0:
        # Pad along the rows
        pad_width = ((0, row_diff), (0, 0))
        processed_spectrogram = np.pad(processed_spectrogram, pad_width, mode='constant', constant_values=0)
    elif row_diff < 0:
        # Crop along the rows
        processed_spectrogram = processed_spectrogram[:original_rows, :]

    if col_diff > 0:
        # Pad along the columns
        pad_width = ((0, 0), (0, col_diff))
        processed_spectrogram = np.pad(processed_spectrogram, pad_width, mode='constant', constant_values=0)
    elif col_diff < 0:
        # Crop along the columns
        processed_spectrogram = processed_spectrogram[:, :original_cols]

    return processed_spectrogram

# Function to denoise the audio (replace this with your denoising logic)
def denoise_audio(input_audio):
    # Dummy denoising logic for demonstration
    return input_audio

def nearest_power_of_2(number):
    power = 1
    while power < number:
        power *= 2
    return power

# List of predefined audio files
audio_files = {
    "Noised Audio 1": "noisy speech/nosied_p237_106.wav",
    "Noised Audio 2": "noisy speech/nosied_p245_281.wav",
    "Noised Audio 3": "noisy speech/nosied_p246_051.wav",
    "Noised Audio 4": "noisy speech/nosied_p246_349.wav",
    "Noised Audio 5": "noisy speech/nosied_p237_271.wav",
    "Noised Audio 6": "noisy speech/nosied_p237_279.wav",
    "Noised Audio 7": "noisy speech/nosied_p241_292.wav",
    "Noised Audio 8": "noisy speech/nosied_p245_191.wav",
    "Noised Audio 9": "noisy speech/nosied_p245_289.wav",
    "Noised Audio 10": "noisy speech/nosied_p241_297.wav",
    "Noised Audio 11": "noisy speech/nosied_p241_312.wav",
    "Noised Audio 12": "noisy speech/nosied_p245_188.wav",
    "Noised Audio 13": "noisy speech/nosied_p260_108.wav",
    "Noised Audio 14": "noisy speech/nosied_p260_173.wav",
    "Noised Audio 15": "noisy speech/nosied_p260_357.wav",
    "Testing audio 1": "noisy speech/nosied_clnsp10.wav",
    "Testing audio 2": "noisy speech/nosied_clnsp22.wav",
    "Testing audio 3": "noisy speech/nosied_clnsp39.wav",
    # Add more audio files as needed
}

# Streamlit app
def main():
    st.title("Speech Denoising with UNET")

    st.subheader("Overview")
    # Introduction
    st.write(
        "Welcome to the Audio Denoising and Reconstruction App! "
        "This app showcases results of the Speech Denoiser Project. You can visualize the waveform, spectrogram of the original and recontrusted audio "
        "and apply a denoising process to reconstruct a cleaner version of the audio. "
        "The denoising process is done using UNET architecture"
    )

    st.subheader("Overview")
    # Introduction
    st.write(
        "Welcome to the Audio Denoising and Reconstruction App! "
        "This app showcases results of the Speech Denoiser Project. You can visualize the waveform, spectrogram of the original and recontrusted audio "
        "and apply a denoising process to reconstruct a cleaner version of the audio. "
        "The denoising process is done using UNET architecture"
    )

    # Select an audio file from the predefined list
    selected_file = st.selectbox("Select an Audio File", list(audio_files.keys()))

    # Display the selected audio file
    st.audio(audio_files[selected_file], format="audio/wav")

    # Convert the selected file to numpy array (replace this with actual data loading)
    audio, _ = librosa.load(audio_files[selected_file])
    spectrogram= extract(audio)
    original_shape = spectrogram.shape

    # Display the original waveform if the checkbox is selected
    if st.checkbox("Show Waveform", key= "noised waveform"):
        st.header("Noised Speech Waveform")
        plot_waveform(audio, title="Noised Speech Waveform")

    # Show the Spectrogram
    if st.checkbox("Show Spectrogram", key='noised spec'):
        st.header("Noised Speech Spectrogram")
        plot_spect(spectrogram, "Noised Speech Spectrogram")

    original_shape = spectrogram.shape

    # Display the original waveform if the checkbox is selected
    if st.checkbox("Show Waveform", key= "noised waveform"):
        st.header("Noised Speech Waveform")
        plot_waveform(audio, title="Noised Speech Waveform")

    # Show the Spectrogram
    if st.checkbox("Show Spectrogram", key='noised spec'):
        st.header("Noised Speech Spectrogram")
        plot_spect(spectrogram, "Noised Speech Spectrogram")


    # Normalizing
    xshape = 256
    y = spectrogram.shape[1]
    y = nearest_power_of_2(number=y)
    min_val = np.min(spectrogram)
    max_val = np.max(spectrogram)
    normalized_spec = normalize(spectrogram, min_val, max_val)

    # Changing shape and loading as numpy array
    test_data = load_numpy(normalized_spec, (256, y))

    # Creating model
    model = create_model((256, y))

    # Button to trigger denoising and reconstruction
    if st.button("Reconstruct"):
        # Denoise and reconstruct the audio
        denoised_audio = model.reconstruct(test_data)

        # Display the denoised waveform
        st.header("Denoised and Reconstructed Audio")
        signal, constructed_spectrogram = convert_spectrogram_to_audio(denoised_audio[0], min_val, max_val)
        signal, constructed_spectrogram = convert_spectrogram_to_audio(denoised_audio[0], min_val, max_val)

        # reshaping to original shape
        constructed_spectrogram = resize_spectrogram(original_shape, constructed_spectrogram)
        # reshaping to original shape
        constructed_spectrogram = resize_spectrogram(original_shape, constructed_spectrogram)
        signal, _ = librosa.effects.trim(signal, top_db=20)


        sf.write('constructed.wav', signal, 22050)
        st.audio('constructed.wav', format="audio/wav")

        
        st.subheader("Reconstructed Audio Waveform")
        plot_waveform(signal, 'Reconstructed Audio Waveform')

    
        st.subheader("Recontructed Audio Spectrogram")
        plot_spect(constructed_spectrogram, 'Contructed Audio after Noise removal')

    
        
        st.subheader("Reconstructed Audio Waveform")
        plot_waveform(signal, 'Reconstructed Audio Waveform')

    
        st.subheader("Recontructed Audio Spectrogram")
        plot_spect(constructed_spectrogram, 'Contructed Audio after Noise removal')

    


if __name__ == "__main__":
    main()
