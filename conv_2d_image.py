from skimage import data
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

image = data.camera()
plt.figure(figsize=(7,7))
plt.imshow(image, cmap='gray')
plt.show()

print(image.shape)
image = image[None, ..., None]
print(image.shape)
image = image.astype(np.float32)/255

conv = tf.keras.layers.Conv2D(kernel_size=(7,7), filters = 1, use_bias=False)
_ = conv(image)

kernel = np.ones((7,7,1,1))*1./49
conv.set_weights([kernel])
blur_image = conv(image).numpy()
plt.figure(figsize=(7,7))
plt.imshow(blur_image[0,:,:,0], cmap='gray')
plt.show()
print(image.shape)
print(blur_image.shape)