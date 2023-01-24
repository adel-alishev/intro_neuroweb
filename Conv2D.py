import numpy as np
signal = np.array([[3,3,2,1,0],
                   [0,0,1,3,1],
                   [3,1,2,2,3],
                   [2,0,0,2,2],
                   [2,0,0,0,1]])
print(signal)
signal = signal.reshape(1,5,5,1).astype(np.float32)
print(signal)
import tensorflow as tf

conv_layer = tf.keras.layers.Conv2D(kernel_size=(3,3), filters=1, use_bias=False)

output = conv_layer(signal)
print(output.numpy().shape)
print(conv_layer.get_weights()[0].shape)

kernel = np.array([[0,1,2],
                   [2,2,0],
                   [0,1,2]])
kernel = kernel.reshape((3,3,1,1)).astype(np.float32)
conv_layer.set_weights([kernel])
print(conv_layer(signal)[0,:,:,0].numpy())
