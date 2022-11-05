import numpy as np
import glob
import os
import sys
import scipy.io.wavfile as wavf
import scipy.signal
import h5py
import json
import librosa
import multiprocessing
import argparse
from keras import *

from keras.layers import Conv2D, Flatten, MaxPooling2D, Conv3D, ReLU
from keras.losses import sparse_categorical_crossentropy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
from keras import losses
import keras.utils
from keras import utils as np_utils
import tensorflow as tf
from keras import layers, initializers, optimizers
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from keras.datasets import cifar10

# Model configuration
batch_size = 100
img_width, img_height, img_num_channels = 227, 227, 1
loss_function = sparse_categorical_crossentropy
no_classes = 10
no_epochs = 1
optimizer = Adam()
validation_split = 0.2
verbosity = 1


# from tensorflow.keras.utils import to_categorical

def build_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(96, kernel_size=(11, 11), input_shape=(227, 227, 1), strides=(4, 4), activation='relu',
                            padding='valid', name='conv1'))

    model.add(layers.MaxPooling2D(pool_size=3, strides=2, name='pool1'))

    model.add(layers.Conv2D(256, kernel_size=(5, 5), activation='relu', padding='valid', name='conv2'))

    model.add(layers.MaxPooling2D(pool_size=3, strides=2, name='pool2'))

    model.add(layers.Conv2D(384, kernel_size=(3, 3), activation='relu', padding='valid', name='conv3'))

    model.add(layers.Conv2D(384, kernel_size=(3, 3), activation='relu', padding='valid', name='conv4'))

    model.add(layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='valid', name='conv5'))

    model.add(layers.MaxPooling2D(pool_size=3, strides=2, name='pool5'))

    model.add(layers.Flatten())
    model.add(layers.Dense(1024, name='fc6', activation='relu'))
    model.add(layers.Dropout(0.5, name="drop6"))
    model.add(layers.Dense(1024, name='fc7', activation='relu'))
    model.add(layers.Dropout(0.5, name="drop7"))
    model.add(layers.Dense(10, name='fc8', activation='softmax'))

    return model


cnn_model = build_model()
cnn_model.summary()

# Compile the cnn_model
cnn_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizers.Adam(learning_rate=0.0005),
                  metrics=["accuracy"])

print("after compile")

train_data = []
train_label = []

test_data = []
test_label = []
print("train_data")
with open('C:\\Users\\trist\\PycharmProjects\\AudioMNIST\\preprocessed_data\\AlexNet_digit_0_train.txt') as f:
    contents = f.readlines()
    # print(contents)
    for line in contents:
        f = h5py.File(line.strip(), 'r')
        train_data.append(f['data'][...])
        train_label.append(f['label'][...])
        # print(train_data)

#print(train_data)
print("test_data")
with open('C:\\Users\\trist\\PycharmProjects\\AudioMNIST\\preprocessed_data\\AlexNet_digit_0_test.txt') as f:
    contents = f.readlines()
    # print(contents)
    for line in contents:
        f = h5py.File(line.strip(), 'r')
        test_data.append(f['data'][...])
        test_label.append(f['label'][...])


# print(train_data)
print("fit")
#
# Fit data to model
# history = cnn_model.fit(train_data, train_label,
#                         batch_size=batch_size,
#                         epochs=1,
#                         verbose=1,
#                         validation_split=0)

score = cnn_model.evaluate(test_data, test_label, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
