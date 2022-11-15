import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import wave
import pylab
from pathlib import Path
from scipy import signal
from scipy.io import wavfile
from sklearn.metrics import confusion_matrix
import itertools
import keras
from keras import layers, optimizers
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy

INPUT_DIR = os.path.join(os.getcwd(), "data")
OUTPUT_DIR = os.path.join(os.getcwd(), "wave")
print(INPUT_DIR)

folders = []
for folder in os.listdir(INPUT_DIR):
    if not os.path.isdir(os.path.join(INPUT_DIR, folder)):
        continue
    folders.append(folder)
print(folders)

path = []
for pat in folders:
    phrase = INPUT_DIR + '\\' + pat
    path.append(phrase)

parent_list = []
for i in range(60):
    parent_list.append(os.listdir(path[i]))
print(parent_list)


IMAGE_HEIGHT = 227
IMAGE_WIDTH = 227
BATCH_SIZE = 32
N_CHANNELS = 3
N_CLASSES = 10

# Make a dataset containing the training spectrograms
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
                                             batch_size=BATCH_SIZE,
                                             validation_split=0.2,
                                             directory=OUTPUT_DIR,
                                             shuffle=True,
                                             color_mode='rgb',
                                             image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                             subset="training",
                                             seed=0)
valid_dataset = tf.keras.preprocessing.image_dataset_from_directory(
                                             batch_size=BATCH_SIZE,
                                             validation_split=0.2,
                                             directory=OUTPUT_DIR,
                                             shuffle=True,
                                             color_mode='rgb',
                                             image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                             subset="validation",
                                             seed=0)


def prepare(ds, augment=False):
    # Define our one transformation
    rescale = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)])
    flip_and_rotate = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)
    ])

    # Apply rescale to both datasets and augmentation only to training
    ds = ds.map(lambda x, y: (rescale(x, training=True), y))
    if augment: ds = ds.map(lambda x, y: (flip_and_rotate(x, training=True), y))
    return ds


train_dataset = prepare(train_dataset, augment=False)
valid_dataset = prepare(valid_dataset, augment=False)




#AudioNet
def build_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(100, kernel_size=(1, 3), input_shape=(227, 227, 3), strides=(1, 1), activation='relu',
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
history = cnn_model.fit(train_dataset, validation_data=valid_dataset)


final_loss, final_acc = cnn_model.evaluate(valid_dataset, verbose=0)
print("Final loss: {0:.6f}, final accuracy: {1:.6f}".format(final_loss, final_acc))


"""
#AlexNet
def build_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(96, kernel_size=(11, 11), input_shape=(227, 227, 3), strides=(4, 4), activation='relu',
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
cnn_model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=optimizers.Adam(learning_rate=0.0001),
                  metrics=["accuracy"])

history = cnn_model.fit(train_dataset, validation_data=valid_dataset)


final_loss, final_acc = cnn_model.evaluate(valid_dataset, verbose=0)
print("Final loss: {0:.6f}, final accuracy: {1:.6f}".format(final_loss, final_acc))
"""

"""
def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.frombuffer(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

for pat in path:
    for filename in os.listdir(pat):
        #print(filename)
        if "wav" in filename:
            file_path = os.path.join(pat, filename)
            file_stem = Path(file_path).stem
            target_dir = f'class_{file_stem[0]}'
            dist_dir = os.path.join(OUTPUT_DIR, target_dir)
            file_dist_path = os.path.join(dist_dir, file_stem)
            if not os.path.exists(file_dist_path + '.png'):
                if not os.path.exists(dist_dir):
                    os.mkdir(dist_dir)
                file_stem = Path(file_path).stem
                sound_info, frame_rate = get_wav_info(file_path)
                pylab.plot(sound_info)
                #pylab.specgram(sound_info, Fs=frame_rate)
                pylab.savefig(f'{file_dist_path}.png')
                pylab.close()
"""

"""
for i in range(5):
    signal_wave = wave.open(os.path.join(path[0], parent_list[0][i]), 'r')
    sample_rate = 16000
    sig = np.frombuffer(signal_wave.readframes(sample_rate), dtype=np.int16)

    plt.figure(figsize=(12,12))
    plot_a = plt.subplot(211)
    plot_a.set_title(parent_list[0][i])
    plot_a.plot(sig)
    plot_a.set_xlabel('sample rate * time')
    plot_a.set_ylabel('energy')

    plot_b = plt.subplot(212)
    plot_b.specgram(sig, NFFT=1024, Fs=sample_rate, noverlap=900)
    plot_b.set_xlabel('Time')
    plot_b.set_ylabel('Frequency')

plt.show()
"""