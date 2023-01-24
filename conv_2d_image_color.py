from skimage import data
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

image = data.coffee()
plt.figure(figsize=(7,7))
plt.imshow(image)
plt.show()

print(image.shape)
image = image[None, ...]
print(image.shape)
image = image.astype(np.float32)/255

conv_layer = tf.keras.layers.Conv2D(kernel_size=(5, 5), filters=3, use_bias=False)
_ = conv_layer(image)

print(conv_layer.get_weights()[0].shape)

kernel = np.zeros((5,5,3,3))

kernel[:, :, 0, 0] = 1/25. # размытие нулевого канала 0, 0 -- означает, что нулевой канал выхода зависит от нулевого канала входа
                           # [:, :, 1, 0] и [:, :, 2, 0] -- остались равными нулю.
kernel[:, :, 1, 1] = 1/25.
kernel[:, :, 2, 2] = 1/25.
conv_layer.set_weights([kernel])

plt.figure(figsize=(10, 10))
plt.imshow(conv_layer(image).numpy()[0])
plt.show()
print(f"Input shape {image.shape}. Output shape {conv_layer(image).numpy().shape}")

kernel = np.zeros((5,5,3,3))

kernel[:, :, 0, 0] = 1/25. # если оставить только эту строчку -- размоем только нулевой канал (красный), а остальные занулим
kernel[:, :, 2, 2] = 1/25.
conv_layer.set_weights([kernel])

plt.figure(figsize=(10, 10))
plt.imshow(conv_layer(image).numpy()[0])
plt.show()
print(f"Input shape {image.shape}. Output shape {conv_layer(image).numpy().shape}")