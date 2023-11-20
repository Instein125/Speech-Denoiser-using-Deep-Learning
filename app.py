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
        plot_spect(spectrogram, 'Contructed Audio after Noise removal')
            
        # apply denormalisation
        denorm_log_spec = denormalize(
            spectrogram, min_val, max_val)
        # log spectrogram -> spectrogram
        spec = librosa.db_to_amplitude(denorm_log_spec)
        # apply Griffin-Lim
        signal = librosa.istft(spec, hop_length=256)
        return signal

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



# Function to denoise the audio (replace this with your denoising logic)
def denoise_audio(input_audio):
    # Dummy denoising logic for demonstration
    return input_audio

# List of predefined audio files
audio_files = {
    "Noised Audio 1": "samples/original/noised_speech_1.wav",
    "Noised Audio 2": "samples/original/noised_speech_2.wav",
    "Noised Audio 3": "samples/original/noised_speech_3.wav",
    "Noised Audio 4": "samples/original/noised_speech_4.wav",
    # Add more audio files as needed
}

# Streamlit app
def main():
    st.title("Speech Denoising with UNET")

    # Select an audio file from the predefined list
    selected_file = st.selectbox("Select an Audio File", list(audio_files.keys()))

    # Display the selected audio file
    st.audio(audio_files[selected_file], format="audio/wav")

    # Convert the selected file to numpy array (replace this with actual data loading)
    audio, _ = librosa.load(audio_files[selected_file])

    # Extract spectrogram
    spectrogram= extract(audio)

    # Normalizing
    min_val = np.min(spectrogram)
    max_val = np.max(spectrogram)
    normalized_spec = normalize(spectrogram, min_val, max_val)

    # Changing shape and loading as numpy array
    test_data = load_numpy(normalized_spec, (256, 512))

    # Creating model
    model = create_model((256, 512))

    # Button to trigger denoising and reconstruction
    if st.button("Reconstruct"):
        # Denoise and reconstruct the audio
        denoised_audio = model.reconstruct(test_data)

        # Display the denoised waveform
        st.header("Denoised and Reconstructed Audio")
        signal = convert_spectrogram_to_audio(denoised_audio[0], min_val, max_val)

        # Display the denoised audio
        # st.audio(signal, format="audio/wav")
        signal, _ = librosa.effects.trim(signal, top_db=20)
        plot_waveform(signal, 'Reconstructed Audio Waveform')
        sf.write('constructed.wav', signal, 22050)
        st.audio('constructed.wav', format="audio/wav")

            # Button to download the denoised audio
        if st.button("Download Cleaned Audio"):
            # Save the denoised audio to a temporary file
            temp_path = "cleaned_audio.wav"
            write(temp_path, 22050, signal)

            # Provide a link to download the file
            st.markdown(f"[Click here to download cleaned audio]({temp_path})")

    # Show the waveform
    "Waveforms"
    plot_waveform(audio, title="Noised Speech Waveform")

    # Show spectrogram
    "Spectrogram"
    plot_spect(spectrogram, "Noised Speech Spectrogram")


if __name__ == "__main__":
    main()
