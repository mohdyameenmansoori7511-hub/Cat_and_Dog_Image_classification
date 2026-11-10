"""Model Comparison and Transfer Learning Setup"""
from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, models


def build_mobilenetv2_classifier(
    input_shape: Tuple[int, int, int] = (160, 160, 3)
) -> tf.keras.Model:
    """Builds a frozen MobileNetV2 transfer learning model for binary classification."""
    inputs = layers.Input(shape=input_shape, name="input_image")
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    base_model.trainable = False
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="binary_output")(x)
    return models.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_Cat_Dog")