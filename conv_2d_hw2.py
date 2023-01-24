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

conv_layer = tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=3, use_bias=False, padding='same')
_ = conv_layer(image)

print(conv_layer.get_weights()[0].shape)

kernel_shaped = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
kernel = np.zeros((3,3,3,3))

kernel[:, :, 0, 0] = kernel_shaped # размытие нулевого канала 0, 0 -- означает, что нулевой канал выхода зависит от нулевого канала входа
                           # [:, :, 1, 0] и [:, :, 2, 0] -- остались равными нулю.
kernel[:, :, 1, 1] = kernel_shaped
kernel[:, :, 2, 2] = kernel_shaped
conv_layer.set_weights([kernel])

sharpen_image = conv_layer(image).numpy()
plt.figure(figsize=(10, 10))
plt.imshow(sharpen_image[0])
plt.show()
print(f"Input shape {image.shape}. Output shape {sharpen_image.shape}")
assert image.shape  == sharpen_image.shape
print("Tests passed")