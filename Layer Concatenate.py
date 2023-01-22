import tensorflow as tf
from tensorflow.keras.layers import Dense
import numpy as np

input_layer = tf.keras.layers.Input(shape=(10, ), name="Input")
dense_layer = Dense(10)

l1_output = Dense(10, name="Layer1")(input_layer)
l2_output = Dense(10, name="Layer2")(input_layer)

l3_output = Dense(10, name="Layer3")(l1_output)
l4_output = Dense(10, name="Layer4")(l1_output)

l5_output = Dense(10, name="Layer5")(l2_output)
l6_output = Dense(10, name="Layer6")(l2_output)

l7_output = tf.keras.layers.Concatenate(name="ConcatLayer")([l3_output, l4_output, l5_output, l6_output])

l8_output = Dense(3, name="Output")(l7_output)

model = tf.keras.Model(inputs=input_layer, outputs=l8_output)
# <YOUR CODE ENDS HERE >

model.summary()
assert model.count_params() == 783, "Wrong params number"
print("Simple tests passed")
tf.keras.utils.plot_model(model, show_shapes=False)