import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import tensorflow as tf

# dir = os.path.abspath(os.curdir)

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
base_dir = Path(os.curdir).parent / "cats_and_dogs_filtered"
train_dir = base_dir / 'train'
validation_dir = base_dir / 'validation'

train_cats_dir = train_dir / 'cats'
train_dogs_dir = train_dir / 'dogs'
validation_cats_dir = validation_dir / 'cats'
validation_dogs_dir = validation_dir / 'dogs'
num_cats_tr = len(list(train_cats_dir.glob("*"))) # .glob("*") создает итератор по всем файлам в директории
num_dogs_tr = len(list(train_dogs_dir.glob("*")))

num_cats_val = len(list(validation_cats_dir.glob("*")))
num_dogs_val = len(list(validation_dogs_dir.glob("*")))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

print('total training cat images:', num_cats_tr)
print('total training dog images:', num_dogs_tr)

print('total validation cat images:', num_cats_val)
print('total validation dog images:', num_dogs_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)

image_path = str(list(validation_cats_dir.glob("*"))[0])
image = plt.imread(image_path)
plt.figure()
plt.imshow(image)
plt.show()
print(image.shape)

image_path = str(list(validation_dogs_dir.glob("*"))[0])
image = plt.imread(image_path)
plt.figure()
plt.imshow(image)
plt.show()
print(image.shape)

BATCH_SIZE = 100 # размер батча -- т.е. количество картинок которые мы считываем за раз
                 # во многом зависит от доступной памяти GPU.
                 # Если ее не хватит (появится такая ошибка) значит нужно уменьшить размер батча или картинки
IMG_SHAPE  = 150

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_image_generator      = ImageDataGenerator(rescale=1./255)
validation_image_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=train_dir, # путь до папки train
                                                           shuffle=True,
                                                           target_size=(IMG_SHAPE,IMG_SHAPE),
                                                           class_mode='binary') # т.к. у нас два класса можем воспользоваться binary

val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                              directory=validation_dir, # путь до папки validation
                                                              shuffle=False,
                                                              target_size=(IMG_SHAPE,IMG_SHAPE),
                                                              class_mode='binary')
# чтобы понять какой класс считается первым а какой вторым воспользуемся полем .class_indices
print(train_data_gen.class_indices)

sample_training_images, sample_labels = next(train_data_gen) # как только мы вополним эту команду, мы считаем 100 картинок с диска
                                                             # и преобразуем их к нужному формату

print(sample_training_images.shape, sample_labels.shape) # 4x мерный тензор и вектор из меток класса

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

show_catsdogs(sample_training_images, sample_labels)
sample_training_images, sample_labels = next(train_data_gen)  # запустив эту клетку несколько раз картинки удут меняться