"""Model Training, Loss Tracking and Early Stopping """
import tensorflow as tf


def get_early_stopping_callback(patience: int = 4) -> tf.keras.callbacks.EarlyStopping:
    """Returns EarlyStopping callback monitoring validation loss and restoring best weights."""
    return tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )
