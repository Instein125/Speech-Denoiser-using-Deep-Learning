import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

# Downsampling block (encoder)
def downsample_block(input_layer, filters, kernel_size, padding='same', activation='relu'):
    conv1 = layers.Conv2D(filters, kernel_size, activation=activation, padding=padding)(input_layer)
    conv1 = layers.Dropout(0.1)(conv1)
    conv1 = layers.Conv2D(filters, kernel_size, activation=activation, padding=padding)(conv1)
    b1 = layers.BatchNormalization()(conv1)
    r1 = layers.ReLU()(b1)
    pool = layers.MaxPooling2D(pool_size=(2, 2))(r1)
    print("Downsample block shape is: " ,conv1.shape)
    return conv1, pool

# Upsampling block
def upsample_block(input_layer, skip_connection, filters, kernel_size, padding='same', activation='relu'):
    up = layers.UpSampling2D(size=(2, 2))(input_layer)
    up = layers.Conv2DTranspose(filters, kernel_size, activation=activation, padding=padding)(up)
    merge = layers.concatenate([up, skip_connection], axis=3)
    conv = layers.Conv2D(filters, 3, activation=activation, padding=padding)(merge)
    conv = layers.Conv2D(filters, 3, activation=activation, padding=padding)(conv)
    print("upsample block shape is: " ,conv.shape)
    return conv

# Define the U-Net architecture
def unet(input_shape):
    inputs = tf.keras.Input(input_shape)

    conv1, pool1 = downsample_block(inputs, 64, (3, 3))
    conv2, pool2 = downsample_block(pool1, 128, (3, 3))
    conv3, pool3 = downsample_block(pool2, 256, (3, 3))
    conv4, pool4 = downsample_block(pool3, 512, (3, 3))

    conv5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)

    conv6 = upsample_block(conv5, conv4, 512, (3, 3))
    conv7 = upsample_block(conv6, conv3, 256, (3, 3))
    conv8 = upsample_block(conv7, conv2, 128, (3, 3))
    conv9 = upsample_block(conv8, conv1, 64, (3, 3))

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv9)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model