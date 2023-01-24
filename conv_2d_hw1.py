import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D
from skimage import data
import matplotlib.pyplot as plt

image = data.camera()
plt.figure(figsize=(7, 7))
plt.imshow(image, cmap="gray")
plt.show()
image = image[None, ..., None].astype(np.float32) / 255.
print(image.shape)
conv_layer = tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=1, use_bias=False, padding='same')
_=conv_layer(image)
kernel = np.array([[1, 0, -1],
          [2, 0, -2],
          [1, 0, -1]])
kernel = kernel.reshape((3,3,1,1))
conv_layer.set_weights([kernel])
print(conv_layer(image)[0,:,:,0].numpy())
detected_lines = conv_layer(image)
plt.figure(figsize=(7, 7))
plt.imshow(np.abs(detected_lines.numpy()[0, :,:, 0]), cmap="gray")
plt.show()
print(f"Input shape {image.shape}. Output shape {conv_layer(image).numpy().shape}")
assert image.shape  == detected_lines.shape
print("Tests passed")