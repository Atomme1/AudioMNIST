import h5py
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import Adam
import deepdish as dd
import tensorflow as tf

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# Model configuration
batch_size = 50
img_width, img_height, img_num_channels = 257, 257, 1
loss_function = sparse_categorical_crossentropy
no_classes = 10
no_epochs = 25
optimizer = Adam()
validation_split = 0.2
verbosity = 1

train_data=[]
train_label=[]
# Load MNIST data
f = h5py.File("C://Users//trist//PycharmProjects//AudioMNIST//preprocessed_data//01//AlexNet_0_01_0.hdf5", 'r')
train_data.append(f['data'])
train_label.append(f['label'])

f = h5py.File("C://Users//trist//PycharmProjects//AudioMNIST//preprocessed_data//01//AlexNet_0_01_1.hdf5", 'r')
train_data.append(f['data'][...])
train_label.append(f['label'][...])


f = h5py.File("C://Users//trist//PycharmProjects//AudioMNIST//preprocessed_data//01//AlexNet_0_01_2.hdf5", 'r')
train_data.append(f['data'][...])
train_label.append(f['label'][...])

f = h5py.File("C://Users//trist//PycharmProjects//AudioMNIST//preprocessed_data//01//AlexNet_0_01_3.hdf5", 'r')
train_data.append(f['data'][...])
train_label.append(f['label'][...])

print(train_data[0])
print(train_label[0])

##tree test
print("tree test")
def h5_tree(val, pre=''):
    items = len(val)
    for key, val in val.items():
        items -= 1
        if items == 0:
            # the last item
            if type(val) == h5py._hl.group.Group:
                print(pre + '└── ' + key)
                h5_tree(val, pre + '    ')
            else:
                print(pre + '└── ' + key + ' (%d)' % len(val))
        else:
            if type(val) == h5py._hl.group.Group:
                print(pre + '├── ' + key)
                h5_tree(val, pre + '│   ')
            else:
                print(pre + '├── ' + key + ' (%d)' % len(val))


with h5py.File("C://Users//trist//PycharmProjects//AudioMNIST//preprocessed_data//01//AlexNet_0_01_0.hdf5", 'r') as hf:
    print(hf)
    h5_tree(hf)

mydata = dd.io.load("C://Users//trist//PycharmProjects//AudioMNIST//preprocessed_data//01//AlexNet_0_01_3.hdf5")
print(mydata.keys())
# print(mydata['data'][0], type(_float_feature(mydata['data'][0])))
print("len(f)")
print(f['data'][...])

f.close()
f = h5py.File("C://Users//trist//PycharmProjects//AudioMNIST//preprocessed_data//01//AlexNet_0_01_1.hdf5", 'r')
input_test = f['data'][...]
label_test = f['label'][...]
f.close()

# Reshape data
#input_train = input_train.reshape((len(input_train), img_width, img_height, img_num_channels))
#input_test = input_test.reshape((len(input_test), img_width, img_height, img_num_channels))

# Determine shape of the data
input_shape = (img_width, img_height, img_num_channels)
#
# # Create the model
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(no_classes, activation='softmax'))
#
# # Display a model summary
# model.summary()
#
# # Compile the model
# model.compile(loss=loss_function,
#               optimizer=optimizer,
#               metrics=['accuracy'])
#
# # Fit data to model
# history = model.fit(train_data, train_label,
#                     batch_size=batch_size,
#                     epochs=no_epochs,
#                     verbose=verbosity,
#                     validation_split=validation_split)

# Generate generalization metrics
# score = model.evaluate(input_test, label_test, verbose=0)
# print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
