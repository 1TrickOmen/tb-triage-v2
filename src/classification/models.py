from tensorflow.keras import layers
import tensorflow as tf


def _set_base_trainability(base_model, trainable_base=False, trainable_fraction=None):
    if trainable_fraction is not None:
        if not 0.0 <= float(trainable_fraction) <= 1.0:
            raise ValueError('trainable_fraction must be between 0.0 and 1.0')
        if float(trainable_fraction) == 0.0:
            base_model.trainable = False
            return

        base_model.trainable = True
        total_layers = len(base_model.layers)
        trainable_layers = max(1, int(round(total_layers * float(trainable_fraction))))
        frozen_until = max(0, total_layers - trainable_layers)
        for layer_index, layer in enumerate(base_model.layers):
            layer.trainable = layer_index >= frozen_until
        return

    base_model.trainable = trainable_base


def build_mobilenetv2(
    input_shape=(256, 256, 3),
    num_classes=2,
    dropout=0.3,
    dense_units=128,
    trainable_base=False,
    trainable_fraction=None,
):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    _set_base_trainability(base_model, trainable_base=trainable_base, trainable_fraction=trainable_fraction)

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    last_conv_layer_name = 'block_13_expand_relu'
    return model, last_conv_layer_name


def build_densenet121(
    input_shape=(256, 256, 3),
    num_classes=2,
    dropout=0.3,
    dense_units=128,
    trainable_base=False,
    trainable_fraction=None,
):
    base_model = tf.keras.applications.DenseNet121(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    _set_base_trainability(base_model, trainable_base=trainable_base, trainable_fraction=trainable_fraction)

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    last_conv_layer_name = 'relu'  # For DenseNet121 Grad-CAM
    return model, last_conv_layer_name
