from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense
from keras import backend as K
import os
import numpy as np
import matplotlib.pyplot as plt


dataset_folder = 'simpsons_dataset'

def load_data(filename, force=False):
  root = os.path.join(filename)
  data_folders = [
    os.path.join(root, d) for d in sorted(os.listdir(root))
    if os.path.isdir(os.path.join(root, d))]
  print(data_folders)
  return data_folders
  
data_folders = load_data(dataset_folder)

train_data_dir = os.path.join(dataset_folder, 'simpsons_trainset')
validation_data_dir = os.path.join(dataset_folder, 'simpson_testset')

img_width, img_height = 64, 64
nb_train_samples = 30000
nb_validation_samples = 990
epochs = 30
batch_size = 32

# Model definition
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()

model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255.0,
    shear_range=0.2,
    zoom_range=0.2,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    fill_mode = 'nearest',
    horizontal_flip=True)

# Only rescaling for validation
valid_datagen = ImageDataGenerator(rescale=1. / 255.0)
 
# Flows the data directly from the directory structure, resizing where needed
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = valid_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)


# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

    # Save weights for further use
model.save_weights('simpsons_weights.h5')
model.save('simpsons_model_trained')
