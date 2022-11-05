import numpy as np
import glob
import os
import sys
import h5py
import json
import librosa

# load the data from txt files
sample_file = 'C:\\Users\\trist\\PycharmProjects\\AudioMNIST\\preprocessed_data\\AlexNet_digit_0_train.txt'

os.chdir(os.path.join(os.getcwd(), "preprocessed_data"))
print(os.getcwd())


# f = h5py.File('./train.hdf5', 'r')
# input_train = f['image'][...]
# label_train = f['label'][...]
# f.close()
# f = h5py.File('./test.hdf5', 'r')
# input_test = f['image'][...]
# label_test = f['label'][...]
# f.close()
train_data = []
train_label = []
with open('C:\\Users\\trist\\PycharmProjects\\AudioMNIST\\preprocessed_data\\AlexNet_digit_0_train.txt') as f:
    contents = f.readlines()
    print(contents)
    for line in contents:
        f = h5py.File(line.strip(), 'r')
        train_data.append(f['data'][...])
        train_label.append(f['label'][...])
        print(train_data)