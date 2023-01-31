import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import image
import numpy as np

VGG19 = tf.keras.applications.VGG19(weights='imagenet')
VGG19.summary()

#подготавливаем четырехмерный тензор

def prepare_image(image, image_size):
    image = tf.image.resize(image, image_size)
    return image[None,...]


from tensorflow.keras.applications.vgg19 import decode_predictions
from tensorflow.keras.applications.vgg19 import preprocess_input
def make_prediction(model, preprocess_input, decode_prediction, image):
    img_size = (model.input_shape[1], model.input_shape[2])
    input_image = prepare_image(image, img_size)
    input_image = preprocess_input(input_image)
    prediction = model.predict(input_image)
    print(decode_predictions(prediction))
    return decode_predictions(prediction)

ball = image.imread('ball1.jpg')
make_prediction(VGG19, preprocess_input, decode_predictions, ball)




