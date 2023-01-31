import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import image
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


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

mobilenet = tf.keras.applications.MobileNetV2(weights='imagenet')
mobilenet.summary()

ball = image.imread('ball.jpg')
make_prediction(mobilenet, preprocess_input, decode_predictions, ball)




