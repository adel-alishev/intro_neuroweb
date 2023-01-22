import tensorflow as tf
from tensorflow.keras.layers import Dense
import numpy as np

input_layer = tf.keras.layers.Input(shape=(10,), name='Input')
l1_output = Dense(10, name = 'Layer1')(input_layer)
output = tf.keras.layers.Add()([input_layer, l1_output])
model = tf.keras.Model(input_layer, output)
model.summary()
assert model.count_params() == 110, "Wrong params number"
print("Simple tests passed")
tf.keras.utils.plot_model(model, show_shapes=True)