import tensorflow as tf
import numpy as np

print(tf.__version__)

# определим наш скромный датасет
celsius = np.array([-10, -40, 10, 20, 36, 5, -12, 14, 36]).astype(np.float32)
fahrenheit = np.array([14., -40., 50., 68., 96.8, 41., 10.4, 57.2, 96.8])

#model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=(1,)))
optimizer = tf.keras.optimizers.Adam(learning_rate=0.5)
loss = tf.keras.losses.MSE

#compile
model.compile(optimizer=optimizer, loss=loss)
history = model.fit(celsius, fahrenheit, epochs=200, verbose=2)
print(history)

print(model.get_weights())

import pandas as pd
history_df = pd.DataFrame(history.history)
print(history_df.head())
import matplotlib.pyplot as plt
plt.plot(history_df.loss)
plt.show()

