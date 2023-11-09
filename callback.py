import keras
import warnings
import numpy as np

class IoUCheckpoint(keras.callbacks.Callback):
    def __init__(self, filepath, monitor='iou', verbose=1, save_best_only=True, mode='max'):
        super(IoUCheckpoint, self).__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.mode = mode
        self.best_iou = -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current_iou = logs.get(self.monitor)
        if current_iou is None:
            warnings.warn(f'Cannot save best model. {self.monitor} is not available.', RuntimeWarning)
        else:
            if current_iou > self.best_iou:
                if self.verbose > 0:
                    print(f"\nEpoch {epoch + 1}: {self.monitor} improved from {self.best_iou} to {current_iou}.")
                self.best_iou = current_iou
                if self.save_best_only:
                    # file_path = self.filepath.format(epoch=epoch + 1, iou=current_iou)
                    self.model.save_weights(self.filepath, overwrite = True)
                    if self.verbose > 0:
                        print(f"\nSaved best model based on {self.monitor} to {self.filepath}")

    

def callbackList(filepath = "/content/drive/My Drive/Colab Notebooks/speech denoiser/training/best_weight.hdf5", verbose=1, save_best_only=True):
    iou_checkpoint = IoUCheckpoint(filepath=filepath, monitor='iou', verbose=verbose, save_best_only=save_best_only)

    # Adding the callback to the list of callbacks during model training
    callbacks = [iou_checkpoint]
    return callbacks
