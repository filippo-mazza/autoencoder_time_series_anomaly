from tensorflow import keras
from tensorflow.keras import layers


model = keras.Sequential(
    [
        layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
        layers.Conv1D(
            filters=64, kernel_size=5, padding="same", strides=2, activation="relu"
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1D(
            filters=32, kernel_size=5, padding="same", strides=2, activation="relu"
        ),
        layers.Conv1DTranspose(
            filters=16, kernel_size=5, padding="same", strides=2, activation="relu"
        ),
        layers.Dropout(rate=0.1),
        layers.Conv1DTranspose(
            filters=8, kernel_size=5, padding="same", strides=2, activation="relu"
        ),
        layers.Conv1DTranspose(filters=1, kernel_size=5, padding="same"),
    ]
)
