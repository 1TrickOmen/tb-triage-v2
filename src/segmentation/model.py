from tensorflow.keras import layers, models


def build_unet(input_shape=(512, 512, 1), num_classes=1):
    inputs = layers.Input(shape=input_shape)

    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)

    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)

    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)

    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
    conv4 = layers.BatchNormalization()(conv4)

    up6 = layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv4)
    merge6 = layers.concatenate([up6, conv3])
    conv6 = layers.Conv2D(256, 3, activation='relu', padding='same')(merge6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv6)
    conv6 = layers.BatchNormalization()(conv6)

    up7 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv6)
    merge7 = layers.concatenate([up7, conv2])
    conv7 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge7)
    conv7 = layers.BatchNormalization()(conv7)
    conv7 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv7)
    conv7 = layers.BatchNormalization()(conv7)

    up8 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv7)
    merge8 = layers.concatenate([up8, conv1])
    conv8 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge8)
    conv8 = layers.BatchNormalization()(conv8)
    conv8 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv8)
    conv8 = layers.BatchNormalization()(conv8)

    outputs = layers.Conv2D(num_classes, 1, activation='sigmoid')(conv8)
    return models.Model(inputs=inputs, outputs=outputs)
