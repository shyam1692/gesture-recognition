"""
Here we train a custom neurla network for the classification of Hand gestures.
"""
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')


import keras
from keras.models import Sequential
from keras.optimizers import RMSprop
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.layers import Dropout, Flatten, Dense, Activation, Conv2D, MaxPooling2D
from keras import applications
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import math
import cv2
import threading
from PIL import Image
import os

#Image Dimensions
img_width, img_height = 320, 240

#Data Directories
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

batch_size = 8
num_classes = 3
epochs = 20


#Loading Training Data
train_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(
                                                        img_width, img_height),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=False)

#Class labels for training data
train_labels = train_generator.classes
train_labels = to_categorical(train_labels, num_classes=num_classes)

print(len(train_generator.filenames))
print(train_generator.class_indices)
print(len(train_generator.class_indices))

#Class labels for test data
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(validation_data_dir,
                                                  target_size=(
                                                      img_width, img_height),
                                                  batch_size=batch_size,
                                                  class_mode='categorical',
                                                  shuffle=False)

# get the class lebels for the training data, in the original order
test_labels = test_generator.classes
# convert the training labels to categorical vectors
test_labels = to_categorical(test_labels, num_classes=num_classes)

print(len(test_generator.filenames))
print(test_generator.class_indices)
print(len(test_generator.class_indices))

'''
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
'''


model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
#model.add(Dense(512, activation='relu', input_shape=(img_width, img_height, 3)))
#model.add(Dense(512))
#model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
'''
# Another network
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
'''
model.summary()

#opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
opt = keras.optimizers.SGD(
    lr=0.0001, momentum=0.9, decay=0.0, nesterov=False)


model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

history = model.fit_generator(train_generator,
                              epochs=epochs,
                              verbose=1,
                              validation_data=test_generator)
(eval_loss, eval_accuracy) = model.evaluate_generator(
    test_generator)

# serialize model to YAML
model_yaml = model.to_yaml()
with open("model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
print("[INFO] Loss: {}".format(eval_loss))

'''
plt.figure(1)

# summarize history for accuracy

plt.subplot(211)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# summarize history for loss

plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('performance.png')
'''
fig1 = plt.figure(1)
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
fig1.savefig('Model_ACC.png')


fig2 = plt.figure(2)

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
fig2.savefig('Model_LOSS.png')