from tensorflow.keras import layers
import tensorflow as tf


def build_mobilenetv2(input_shape=(256, 256, 3), num_classes=2, dropout=0.3, dense_units=128, trainable_base=False):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = trainable_base

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    last_conv_layer_name = 'block_13_expand_relu'
    return model, last_conv_layer_name


def build_densenet121(input_shape=(256, 256, 3), num_classes=2, dropout=0.3, dense_units=128, trainable_base=False):
    base_model = tf.keras.applications.DenseNet121(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = trainable_base

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    last_conv_layer_name = 'relu'  # For DenseNet121 Grad-CAM
    return model, last_conv_layer_name
