import h5py
from keras import *

from keras.layers import Conv2D, Flatten, MaxPooling2D, Conv3D, ReLU
from keras.losses import sparse_categorical_crossentropy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD, Adam
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

def build_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(100, kernel_size=(1, 3), input_shape=(227, 227, 1), strides=(1, 1), activation='relu',
                            padding='valid', name='conv1'))

    model.add(layers.MaxPooling2D(pool_size=(1, 3), strides=2, name='pool1'))

    model.add(layers.Conv2D(64, kernel_size=(1, 3), activation='relu', padding='valid', name='conv2'))

    model.add(layers.MaxPooling2D(pool_size=(1, 2), strides=2, name='pool2'))

    model.add(layers.Conv2D(128, kernel_size=(1, 3), strides=(1, 1), activation='relu', padding='valid', name='conv3'))

    model.add(layers.MaxPooling2D(pool_size=(1, 2), strides=2, name='pool3'))

    model.add(layers.Conv2D(128, kernel_size=(1, 3), strides=(1, 1), activation='relu', padding='valid', name='conv4'))

    model.add(layers.MaxPooling2D(pool_size=(1, 2), strides=2, name='pool4'))

    model.add(layers.Conv2D(128, kernel_size=(1, 3), strides=(1, 1), activation='relu', padding='valid', name='conv5'))

    model.add(layers.MaxPooling2D(pool_size=(1, 2), strides=2, name='pool5'))

    model.add(layers.Conv2D(128, kernel_size=(1, 3), strides=(1, 1), activation='relu', padding='valid', name='conv6'))

    model.add(layers.MaxPooling2D(pool_size=(1, 2), strides=2, name='pool6'))

    #model.add(layers.Flatten())
    model.add(layers.Dense(1024, name='fc7'))
    model.add(layers.Dropout(0.5, name="drop7"))
    model.add(layers.Dense(512, name='fc8'))
    model.add(layers.Dropout(0.5, name="drop8"))
    model.add(layers.Dense(512, name='fc9'))

    return model


cnn_model = build_model()
cnn_model.summary()

# Compile the cnn_model
cnn_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizers.Adam(learning_rate=0.0001),
                  metrics=["accuracy"])

print("after compile")

train_data = []
train_label = []

test_data = []
test_label = []
print("train_data")
with open('C:\\Users\\rouxe\\PycharmProjects\\AudioMNIST\\preprocessed_data\\AudioNet_digit_0_train.txt') as f:
    contents = f.readlines()
    for line in contents:
        h5f = h5py.File(line, 'r')
        train_data.append(h5f['data'][...])
        train_label.append(h5f['label'][...])
        # print(train_data)

print("test_data")
with open('C:\\Users\\rouxe\\PycharmProjects\\AudioMNIST\\preprocessed_data\\AudioNet_digit_0_test.txt') as f:
    contents = f.readlines()
    for line in contents:
        h5f = h5py.File(line, 'r')
        test_data.append(h5f['data'][...])
        test_label.append(h5f['label'][...])


print("fit")

# Fit data to model
# history = cnn_model.fit(train_data, train_label,
#                         batch_size=batch_size,
#                         epochs=1,
#                         verbose=1,
#                         validation_split=0)

score = cnn_model.evaluate(test_data, test_label, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
