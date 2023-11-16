import os
import numpy as np
from unet import UNET
from callback import callbackList

x_train_dir = '/content/drive/MyDrive/Colab Notebooks/speech denoiser/x_train_noised_speech'
y_train_dir = '/content/drive/MyDrive/Colab Notebooks/speech denoiser/y_train_clean_audio'

SHAPE=(256, 128)
LEARNING_RATE = 0.001
BATCH_SIZE = 10
EPOCHS = 10


# getting the maximum shape of the numpy array
def get_max_shape(dir):
    max_y = 0
    max_x = 0
    for filename in os.listdir(dir):
        if filename.endswith(".npy"):
            file_path = os.path.join(dir, filename)

            # Load the .npy file and append it to the x_train list
            spectrogram = np.load(file_path)
            if spectrogram.shape[1] > max_y:
                max_y = spectrogram.shape[1]

            if spectrogram.shape[0] > max_x:
                max_x = spectrogram.shape[0]

    return (max_x, max_y)

# load numpy spectrograms
def load_array(dir, final_shape):
    # Initialize an empty list to store the loaded spectrograms
    arr = []
    # Get the list of files in the directory and sort them alphabetically
    file_list = sorted(os.listdir(dir))
    # Iterate through each file in the directory
    for filename in file_list:
        if filename.endswith(".npy"):
            file_path = os.path.join(dir, filename)
            print(file_path)

            # Load the .npy file and append it to the x_train list
            spectrogram = np.load(file_path)
            # Pad the spectrogram to match the desired final shape
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

    return arr


def train(x_train, y_train, learning_rate, batch_size, epochs, callbacks):
    model = UNET(
        input_shape=(256, 256, 1),
        conv_filters=(64, 128, 256, 512),
        conv_kernels=(3, 3),
    )
    model.summary()
    model.compile(learning_rate)
    history=model.train(x_train, y_train, batch_size, epochs, callbacks)
    return model, history


if __name__ == "__main__":
    shape=get_max_shape(x_train_dir)
    print(shape)
    x_train = load_array(x_train_dir, SHAPE)
    y_train = load_array(y_train_dir, SHAPE)
    callbacks = callbackList()
    model, history = train(x_train, 
                           y_train, 
                           LEARNING_RATE, 
                           BATCH_SIZE, 
                           EPOCHS, 
                           callbacks)
    model.save("/content/drive/MyDrive/Colab Notebooks/speech denoiser")