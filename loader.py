import numpy as np
import os


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

def get_array(dir, final_shape):
    # Initialize an empty list to store the loaded spectrograms
    arr = []
    # Iterate through each file in the directory
    for filename in os.listdir(dir):
        if filename.endswith(".npy"):
            file_path = os.path.join(dir, filename)

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

    return arr