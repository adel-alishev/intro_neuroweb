import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import image
import numpy as np
from tensorflow.keras.applications.resnet50 import decode_predictions
from tensorflow.keras.applications.resnet50 import preprocess_input


def prepare_image(image, image_size):
    image = tf.image.resize(image, image_size)
    return image[None,...]
def make_prediction(model, preprocess_input, decode_prediction, image):
    img_size = (model.input_shape[1], model.input_shape[2])
    input_image = prepare_image(image, img_size)
    input_image = preprocess_input(input_image)
    prediction = model.predict(input_image)
    print(decode_predictions(prediction))
    return decode_predictions(prediction)

resnet = tf.keras.applications.ResNet50(weights='imagenet')
resnet.summary()

img = image.imread('image1.jpg')
plt.imshow(img)
plt.show()
make_prediction(resnet, preprocess_input, decode_predictions, img)




