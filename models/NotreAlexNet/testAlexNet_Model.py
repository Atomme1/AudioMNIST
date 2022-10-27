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

from keras.layers import Conv2D, Flatten, MaxPooling2D, Conv3D
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import keras
import keras.utils
from keras import utils as np_utils
#yo
# from tensorflow.keras.utils import to_categorical

cnn_model = Sequential()
cnn_model.add(Conv2D(filters=96, input_shape=(224, 224, 3), kernel_size=(11, 11), strides=(4, 4), padding='valid'))
cnn_model.add(Activation('relu'))
# Max Pooling
cnn_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

# 2nd Convolutional Layer
cnn_model.add(Conv2D(filters=256, kernel_size=(11, 11), strides=(1, 1), padding='valid'))
cnn_model.add(Activation('relu'))
# Max Pooling
cnn_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

# 3rd Convolutional Layer
cnn_model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
cnn_model.add(Activation('relu'))

# 4th Convolutional Layer
cnn_model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
cnn_model.add(Activation('relu'))

# 5th Convolutional Layer
cnn_model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
cnn_model.add(Activation('relu'))
# Max Pooling
cnn_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

# Passing it to a Fully Connected layer
cnn_model.add(Flatten())
# 1st Fully Connected Layer
cnn_model.add(Dense(4096, input_shape=(224 * 224 * 3,)))
cnn_model.add(Activation('relu'))
# Add Dropout to prevent overfitting
cnn_model.add(Dropout(0.4))

# 2nd Fully Connected Layer
cnn_model.add(Dense(4096))
cnn_model.add(Activation('relu'))
# Add Dropout
cnn_model.add(Dropout(0.4))

# 3rd Fully Connected Layer
cnn_model.add(Dense(1000))
cnn_model.add(Activation('relu'))
# Add Dropout
cnn_model.add(Dropout(0.4))

# Output Layer
cnn_model.add(Dense(17))
cnn_model.add(Activation('softmax'))

cnn_model.summary()

# Compile the cnn_model
cnn_model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy"])