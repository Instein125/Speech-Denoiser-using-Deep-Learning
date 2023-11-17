import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as k
import matplotlib.pyplot as plt
import os
import pickle

class UNET:
    """
    Unet represents a Deep Convolutional encoder decoder architecture
    with skip connection.
    """
    def __init__(self,
                 input_shape,
                 conv_filters,
                 conv_kernels,):
        self.input_shape = input_shape # [256, 256, 1]
        self.conv_filters = conv_filters # [64, 1280, 256, 512]
        self.conv_kernels = conv_kernels # (3, 3)    


        self.model = None

        self._model_input = None

        self._build()

    def summary(self):
        self.model.summary()

    def compile(self, learning_rate=0.001):
        optimizer = Adam(learning_rate=learning_rate)
        binary_loss = tf.keras.losses.BinaryCrossentropy()
        self.model.compile(optimizer=optimizer,
                           loss=binary_loss,
                           metrics=[self.iou,])
        
    def train(self, x_train, y_train, batch_size, num_epochs, callbacks):
        history=self.model.fit(x_train,
                       y_train,
                       batch_size=batch_size,
                       epochs=num_epochs,
                       callbacks=callbacks,
                       validation_split=0.2,
                       )
        return history
    
    def train_generator(self, train_generator, length, batch_size, num_epochs, callbacks):
        history=self.model.fit_generator(
            generator = train_generator,
            steps_per_epoch=length // batch_size,
                       epochs=num_epochs,
                       callbacks=callbacks,
                       validation_split=0.2,
                       )
        return history
        
    def save(self, save_folder):
        self._create_folder_if_it_doesnt_exist(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    # For predicitons
    def reconstruct(self, spectrograms):
        predictions = self.model.predict(spectrograms)
        return predictions
    
    # Plots the graph for training and validation loss and iou score
    def plot_loss(self, history):
        plt.plot(history.history['loss'],label="loss")
        plt.plot(history.history['val_loss'],label="validation loss")
        plt.legend()
        plt.show()

    def plot_iou(self, history):
        plt.plot(history.history['iou'],label="iou")
        plt.plot(history.history['val_iou'],label="validation iou")
        plt.legend()
        plt.show()

    # Calculates the Iou score matrics    
    def iou(self, y_true, y_pred, smooth = 1):
        y_true = k.flatten(y_true)
        y_pred = k.flatten(y_pred)
        intersection = k.sum(y_true*y_pred)
        union = k.sum(y_true)+k.sum(y_pred)-intersection
        iou_score = (intersection+smooth)/(union+smooth)
        return iou_score
    
    def _create_folder_if_it_doesnt_exist(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _save_parameters(self, save_folder):
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
        ]
        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "weights.h5")
        self.model.save_weights(save_path)

    def _build(self):
        self._build_unet()

    # Define the U-Net architecture
    def _build_unet(self):
        inputs = tf.keras.Input(self.input_shape)

        conv1, pool1 = self._downsample_block(inputs, 64, (3, 3))
        conv2, pool2 = self._downsample_block(pool1, 128, (3, 3))
        conv3, pool3 = self._downsample_block(pool2, 256, (3, 3))
        conv4, pool4 = self._downsample_block(pool3, 512, (3, 3))

        conv5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)

        conv6 = self._upsample_block(conv5, conv4, 512, (3, 3))
        conv7 = self._upsample_block(conv6, conv3, 256, (3, 3))
        conv8 = self._upsample_block(conv7, conv2, 128, (3, 3))
        conv9 = self._upsample_block(conv8, conv1, 64, (3, 3))

        outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv9)

        self.model = models.Model(inputs=inputs, outputs=outputs, name = 'Unet')
    
    # Downsampling block (encoder)
    def _downsample_block(self, input_layer, filters, kernel_size, padding='same', activation='relu'):
        conv1 = layers.Conv2D(filters, kernel_size, activation=activation, padding=padding)(input_layer)
        conv1 = layers.Dropout(0.1)(conv1)
        conv1 = layers.Conv2D(filters, kernel_size, activation=activation, padding=padding)(conv1)
        b1 = layers.BatchNormalization()(conv1)
        r1 = layers.ReLU()(b1)
        pool = layers.MaxPooling2D(pool_size=(2, 2))(r1)
        print("Downsample block shape is: " ,conv1.shape)
        return conv1, pool

    # Upsampling block
    def _upsample_block(self, input_layer, skip_connection, filters, kernel_size, padding='same', activation='relu'):
        up = layers.UpSampling2D(size=(2, 2))(input_layer)
        up = layers.Conv2DTranspose(filters, kernel_size, activation=activation, padding=padding)(up)
        merge = layers.concatenate([up, skip_connection], axis=3)
        conv = layers.Conv2D(filters, 3, activation=activation, padding=padding)(merge)
        conv = layers.Conv2D(filters, 3, activation=activation, padding=padding)(conv)
        print("upsample block shape is: " ,conv.shape)
        return conv
    

if __name__ == "__main__":
    unet_model = UNET(
        input_shape=(256, 256, 1),
        conv_filters=(64, 128, 256, 512),
        conv_kernels=(3, 3),
    )
    unet_model.summary()