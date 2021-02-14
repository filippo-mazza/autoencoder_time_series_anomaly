"""
Autoencoder model in Keras 
inspired by the keras official documentation
"""


import numpy as np
import pandas as pd
from tensorflow import keras

from matplotlib import pyplot as plt

DATAFILE = '/datasets/time_series_testing.csv'
TIME_STEPS = 30
TRAIN_EPOCHS = 50


df = pd.read_csv(DATAFILE , parse_dates=True, index_col="time")

# Normalization 
df_training = (df - df.mean()) / df.std()



# Here we generate the sequences
def _sequences(vals, time_steps):
    out = list()
    for i in range(len(vals) - time_steps):
        out.append(vals[i : (i + time_steps)])
    return np.stack(out)


x_train = _sequences(df.values, TIME_STEPS)


## Get the model
import models
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss="mse")
model.summary()



history = model.fit(
    x_train,
    x_train,
    epochs=TRAIN_EPOCHS,
    batch_size=64,
    validation_split=0.15,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, mode="min")
    ],
)


plt.plot(history.history["loss"], label="Train loss")
plt.plot(history.history["val_loss"], label="Val loss")
plt.legend()
plt.show()



x_train_pred = model.predict(x_train)
