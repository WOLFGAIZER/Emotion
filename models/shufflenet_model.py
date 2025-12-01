"""
Improved ShuffleNetV2 for higher accuracy (Driver Monitoring)
- Stronger backbone
- width_multiplier default = 1.0 (instead of 0.5)
- Deeper stages
- SpatialDropout
- Better normalization
"""

from tensorflow.keras import layers, models
from tensorflow.keras.layers import (
    Input, Conv2D, DepthwiseConv2D, BatchNormalization,
    ReLU, Add, GlobalAveragePooling2D, Dense, Dropout, SpatialDropout2D
)
from tensorflow.keras.models import Model
import tensorflow as tf
import os


def channel_shuffle(x, groups):
    height = tf.shape(x)[1]
    width = tf.shape(x)[2]
    channels = x.shape[-1]
    if channels is None:
        raise ValueError("Channels must be known")
    channels_per_group = channels // groups

    x = tf.reshape(x, [-1, height, width, groups, channels_per_group])
    x = tf.transpose(x, [0, 1, 2, 4, 3])
    x = tf.reshape(x, [-1, height, width, channels])
    return x


def conv_bn_relu(x, filters, kernel_size=3, strides=1):
    x = Conv2D(filters, kernel_size, strides=strides, padding="same", use_bias=False)(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = ReLU()(x)
    return x


def depthwise_conv_bn(x, kernel_size=3, strides=1):
    x = DepthwiseConv2D(kernel_size, strides=strides, padding="same", use_bias=False)(x)
    x = BatchNormalization(momentum=0.9)(x)
    return x


def shufflenet_unit(x, out_channels, strides=1, groups=2):
    in_channels = x.shape[-1]

    if strides == 1:
        assert in_channels % 2 == 0
        half = in_channels // 2
        x1 = layers.Lambda(lambda z: z[:, :, :, :half])(x)
        x2 = layers.Lambda(lambda z: z[:, :, :, half:])(x)

        y = conv_bn_relu(x2, out_channels // 2, kernel_size=1)
        y = depthwise_conv_bn(y, kernel_size=3)
        y = conv_bn_relu(y, out_channels // 2, kernel_size=1)

        out = layers.Concatenate(axis=-1)([x1, y])
        out = layers.Lambda(lambda z: channel_shuffle(z, groups))(out)
        return out

    else:
        y1 = depthwise_conv_bn(x, kernel_size=3, strides=2)
        y1 = conv_bn_relu(y1, out_channels // 2, kernel_size=1)

        y2 = conv_bn_relu(x, out_channels // 2, kernel_size=1)
        y2 = depthwise_conv_bn(y2, kernel_size=3, strides=2)
        y2 = conv_bn_relu(y2, out_channels // 2, kernel_size=1)

        out = layers.Concatenate(axis=-1)([y1, y2])
        out = layers.Lambda(lambda z: channel_shuffle(z, groups))(out)
        return out


def build_shufflenetv2(input_shape=(96, 96, 1), num_classes=4, width_multiplier=1.0):
    inp = Input(shape=input_shape)

    # Stronger stem
    x = conv_bn_relu(inp, int(32 * width_multiplier), kernel_size=3, strides=2)
    x = depthwise_conv_bn(x)
    x = conv_bn_relu(x, int(32 * width_multiplier), kernel_size=1)

    # Deeper stages
    stage_channels = [
        int(64 * width_multiplier),
        int(128 * width_multiplier),
        int(192 * width_multiplier),
        int(256 * width_multiplier),
    ]

    # Stage 1
    x = shufflenet_unit(x, stage_channels[0], strides=2)
    for _ in range(2):
        x = shufflenet_unit(x, stage_channels[0], strides=1)

    # Stage 2
    x = shufflenet_unit(x, stage_channels[1], strides=2)
    for _ in range(3):
        x = shufflenet_unit(x, stage_channels[1], strides=1)

    # Stage 3
    x = shufflenet_unit(x, stage_channels[2], strides=2)
    for _ in range(3):
        x = shufflenet_unit(x, stage_channels[2], strides=1)

    # Stronger head
    x = GlobalAveragePooling2D()(x)
    x = layers.Reshape((1, 1, -1))(x)      # valid Keras reshape
    x = SpatialDropout2D(0.2)(x)
    x = layers.Flatten()(x)   
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)

    out = Dense(num_classes, activation="softmax")(x)

    return Model(inputs=inp, outputs=out, name="ShuffleNetV2_Enhanced")


def load_emotion_weights(model, weights_path):
    if not os.path.exists(weights_path):
        print("[WARN] Weights not found:", weights_path)
        return False
    try:
        model.load_weights(weights_path)
        print("[INFO] Loaded:", weights_path)
        return True
    except Exception as e:
        print("[ERROR] Load failed:", e)
        return False
