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
import re

from sklearn.model_selection import train_test_split

from keras.layers import Conv2D, Flatten, MaxPooling2D, Conv3D, ReLU
from keras.losses import sparse_categorical_crossentropy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
from keras import losses
import keras.utils
from keras.utils import to_categorical
from keras import utils as np_utils
import tensorflow as tf
from keras import layers, initializers, optimizers
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from keras.datasets import cifar10
from tqdm import tqdm

# Model configuration
batch_size = 100
img_width, img_height, img_num_channels = 227, 227, 1
data_shape = (1, img_width, img_height, img_num_channels)
input_shape = (img_width, img_height, img_num_channels)
loss_function = 'categorical_crossentropy'
no_classes = 10
no_epochs = 3
learning_rate = 0.001
optimizer = Adam(lr=learning_rate)
validation_split = 0.2
verbosity = 1
output = 'digits'    # can be 'digits' or 'sex'
#PREPROCESSDATAPATH = 'D:\\AudioMNIST\\preprocessed_data'

def build_model(input_shape):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(96, kernel_size=(11, 11), input_shape=input_shape, strides=(4, 4), activation='relu',
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


def model_summary(model):
    return model.summary()


def model_compile(model):
    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])
    return model


def model_fit(model, input_train, label_train):
    return model.fit(input_train, label_train, batch_size=batch_size, epochs=no_epochs, verbose=verbosity)


# def import_h5py(path):
#     h5pyfile = h5py.File(path, 'r')
#     data = h5pyfile['data'][...]
#     label = h5pyfile['label'][...]
#     del h5pyfile
#     return data, label


# def reshape_data(data, shape):
#     return data.reshape(shape)

# def find_expressions(text, expression):
#     return True if re.search(expression, text) else False

# def is_a_hdf5_file(text, expression = '.hdf5$'):
#     return find_expressions(text, expression)

# def is_a_AlexNet_file(text, expression='AlexNet'):
#     return find_expressions(text, expression)

# def get_list_of_the_files(path, expression='AlexNet'):
#     return [joinPaths(path,subFolder) for subFolder in os.listdir(path) if find_expressions(subFolder, expression) and is_a_hdf5_file(subFolder)]

# def get_sub_folders(path):
#     return os.listdir(path)

# def browse_the_file_tree(path, list_of_file_paths=[]):
#     for subfolder in get_sub_folders(path):
#         path_subfolder = joinPaths(path, subfolder)
#         if is_this_a_folder(path_subfolder):
#             browse_the_file_tree(joinPaths(path, subfolder),list_of_file_paths)
#         if is_a_hdf5_file(path_subfolder) and is_a_AlexNet_file(path_subfolder): 
#             list_of_file_paths.append(path_subfolder)  
#     return list_of_file_paths

# def is_this_a_folder(path):
#     return True if os.path.isdir(path) else False


# def extract_data_and_label_AlexNet(list_of_files):
#     for file in tqdm(list_of_files):
#         data_and_label = import_h5py(file)
#         reshaped_data = reshape_data(data_and_label[0], data_shape)
#         if 'data' not in locals():
#             data = reshaped_data
#             label = data_and_label[1]
#         else:
#             data = stack_np_data(data, reshaped_data)
#             label = stack_np_data(label, data_and_label[1])
#     return data, label


# def joinPaths(path,file):
#     return os.path.join(path, file)

# def stack_np_data(data1, data2):
#     return np.vstack((data1, data2))

# def get_list_path_train_test_from_txt_file(path_txt_file):
#     with open(path_txt_file) as f:
#         read = f.read().splitlines()
#         f.close()
#         return read

def get_digits_or_sex(labels_lists, output):
    return labels_lists[:, 0 if output == 'digits' else 1]

def digit_list_to_one_hot_list(digits):
    return to_categorical(digits)
    
def extract(path_txt_file):
    data = []
    label = []
    with open(path_txt_file) as f:
        contents = f.read().splitlines()
        for file in tqdm(contents):
            f = h5py.File(file, 'r')
            data.append(f['data'][...])
            label.append(f['label'][...])
        data = np.array(data).reshape(len(data),img_width, img_height, img_num_channels)
        label = np.array(label).reshape(len(label),2)
    f.close()
    return data, label
       

#testpathfolder='D:\\AudioMNIST\\preprocessed_data\\02'
# testpathfile='D:\\AudioMNIST\\preprocessed_data\\12\\AlexNet_0_12_0.hdf5'
trainpathtxt = 'D:\\AudioMNIST\\preprocessed_data\\AlexNet_digit_0_train.txt'
testpathtxt = 'D:\\AudioMNIST\\preprocessed_data\\AlexNet_digit_0_test.txt'

#list_files = get_list_of_the_files(testpathfolder)
#list_files = browse_the_file_tree(PREPROCESSDATAPATH)
# list_train_files = get_list_path_train_test_from_txt_file(trainpathtxt)
# list_test_files = get_list_path_train_test_from_txt_file(testpathtxt)

# print("number of train elements : {}\nnumber of test elements : {}".format(len(list_train_files), len(list_test_files)))

input_train, label_train = extract(trainpathtxt)
input_test, label_test = extract(testpathtxt)

# input_train, label_train = extract_data_and_label_AlexNet(list_train_files)
# input_test, label_test = extract_data_and_label_AlexNet(list_test_files)
# print('data shape : {} \nlabel shape : {}'.format(datatrainlist.shape, labeltrainlist.shape))

label_train = get_digits_or_sex(label_train, output)
label_test = get_digits_or_sex(label_test, output)
# print('data shape : {} \ndigits shape : {}'.format(datalist.shape, labellist.shape))

label_train = digit_list_to_one_hot_list(label_train)
label_test = digit_list_to_one_hot_list(label_test)
# print('data shape : {} \ndigits encode shape : {}'.format(datalist.shape, labellist.shape))

# input_train, input_test, label_train, label_test = train_test_split(datalist, labellist, test_size=0.20, random_state=42)
print('input train shape : {} \ninput test shape : {} \nlabel train shape : {} \nlabel test shape : {}'.format(input_train.shape, input_test.shape, label_train.shape, label_test.shape))

f = build_model(input_shape)
model_summary(f)
f = model_compile(f)
model_fit(f, input_train, label_train)

score = f.evaluate(input_test, label_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

#300 s per epoch
