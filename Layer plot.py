import tensorflow as tf
from tensorflow.keras.layers import Dense
import numpy as np

model = tf.keras.Sequential()
model.add(Dense(4, input_shape=(10,),name = 'Dense_17'))
model.add(Dense(8))
model.add(Dense(16))
model.add(Dense(3, activation = 'Softmax'))
model.summary()
tf.keras.utils.plot_model(model, show_shapes=False)

output = model(np.ones((3, 10)))
assert np.allclose(output.numpy().sum(1),  np.ones(3)), "Did you forget softmax in the last layer?"
assert model.count_params() == 279, "Wrong params number"
print("Simple tests passed")