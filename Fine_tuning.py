import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
# dir = os.path.abspath(os.curdir)
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
zip_dir = os.curdir
base_dir = Path(zip_dir).parent / "cats_and_dogs_filtered"
train_dir = base_dir / 'train'
validation_dir = base_dir / 'validation'

train_cats_dir = train_dir / 'cats'
train_dogs_dir = train_dir / 'dogs'
validation_cats_dir = validation_dir / 'cats'
validation_dogs_dir = validation_dir / 'dogs'
num_cats_tr = len(list(train_cats_dir.glob("*")))
num_dogs_tr = len(list(train_dogs_dir.glob("*")))

num_cats_val = len(list(validation_cats_dir.glob("*")))
num_dogs_val = len(list(validation_dogs_dir.glob("*")))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

BATCH_SIZE = 100
IMG_SHAPE = 150

image_data_train = ImageDataGenerator(preprocessing_function = preprocess_input)
validation_image_generator = ImageDataGenerator(preprocessing_function = preprocess_input)
train_data_gen = image_data_train.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_SHAPE,IMG_SHAPE),
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                              directory=validation_dir,
                                                              shuffle=False,
                                                              target_size=(IMG_SHAPE,IMG_SHAPE),
                                                              class_mode='binary')

# sample_training_images, sample_labels = next(train_data_gen)
# _val_images, _val_labels = next(val_data_gen)
# next(val_data_gen)[1]
def show_catsdogs(images, labels, predicted_labels=None):
    names = {0: "Cat", 1: "Dog"}
    plt.figure(figsize=(10,10))
    for i in range(16):
        plt.subplot(4,4, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.gray)
        if predicted_labels is not None:
            title_obj = plt.title(f"Real: {names[labels[i]]}. Pred: {names[predicted_labels[i]]}")
            if labels[i] != predicted_labels[i]:
                plt.setp(title_obj, color='r')
        else:
            plt.title(f"Real label: {names[labels[i]]}")
    plt.show()

IMG_SHAPE = (150,150,3)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,#не загружать последний слой сети
                                               weights='imagenet')
base_model.trainable = False #замораживаем всю модель
#base_model.summary()

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
from livelossplot.tf_keras import PlotLossesCallback
# # указанными параметрами спустя 30 эпох вы увидите точность около 75%.
# # если вас что-то насторожит в графиках, запомните это, это нормально :)
EPOCHS = 30
history = model.fit(
    train_data_gen,
    epochs=EPOCHS,
    validation_data=val_data_gen,
    callbacks=[PlotLossesCallback()]) # мы добавили коллбек для отрисовки прогресса


# shuffle_val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
#                                                               directory=validation_dir,
#                                                               shuffle=True,
#                                                               target_size=(IMG_SHAPE,IMG_SHAPE),
#                                                               class_mode='binary')
# sample_validation_images, sample_validation_labels = next(shuffle_val_data_gen)
# predicted = model.predict(sample_validation_images).flatten()
# classes = []
# for i in predicted:
#   if i>0.5:
#     classes.append(1)
#   else: classes.append(0)