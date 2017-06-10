import numpy as np
import struct
import matplotlib.pyplot as plt


def decode_idx3(dataset_file):
    # read binary data
    bin_data = open(dataset_file, 'rb').read()

    # read headers, the order is: magic number, number of images, height, width
    offset = 0
    fmt_header = '>iiii'
    magic_num, img_num, rows, cols = \
        struct.unpack_from(fmt_header, bin_data, offset)
    print 'magic_num: %d, img_num: %d, image size: %d*%d' % \
          (magic_num, img_num, rows, cols)

    #decode data
    img_size = rows * cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(img_size) + 'B'
    images = np.empty((img_num, rows, cols))
    for i in range(img_num):
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset))\
            .reshape((rows, cols))
        offset += struct.calcsize(fmt_image)
        if (i + 1) % 10000 == 0:
            print 'Decoded %d' % (i + 1)
    return images


def decode_idx1(label_file):
    # read binary data
    bin_data = open(label_file, 'rb').read()

    # read headers, the order is: magic number, number of images, height, width
    offset = 0
    fmt_header = '>ii'
    magic_num, img_num = \
        struct.unpack_from(fmt_header, bin_data, offset)
    print 'magic_num: %d, img_num: %d' % (magic_num, img_num)

    #decode data
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(img_num)
    for i in range(img_num):
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
        if (i + 1) % 10000 == 0:
            print 'Decoded %d' % (i + 1)
    return labels



train_data_file = 'train-images.idx3-ubyte'
train_labels_file = 'train-labels.idx1-ubyte'
test_data_file = 't10k-images.idx3-ubyte'
test_labels_file = 't10k-labels.idx1-ubyte'

train_dataset = decode_idx3(train_data_file)
train_labels = decode_idx1(train_labels_file)
test_dataset = decode_idx3(test_data_file)
test_labels = decode_idx1(test_labels_file)


def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

# Preprocessing
# randomize both dataset and labels
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)


# Preprocessing
# Turn pixel to binary
train_dataset = (train_dataset[:,:,:] > 127) * 255.0
test_dataset = (test_dataset[:,:,:] > 127) * 255.0


# Preprocessing
num_labels = 10

# reshape labels to [0.0, 1.0, ..., 0.0]
train_labels = np.arange(num_labels) == train_labels[:, None]
test_labels = np.arange(num_labels) == test_labels[:, None]


# Preprocessing
# Zero mean and Normalize
img_range = 255.0;
train_dataset = 1.0 * (train_dataset - img_range / 2) / img_range
test_dataset = 1.0 * (test_dataset - img_range / 2) / img_range


# Preprocessing
# split test dataset to valid and test two parts
valid_dataset = test_dataset[:5000]
valid_labels = test_labels[:5000]
test_dataset = test_dataset[5000:10000]
test_labels = test_labels[5000:10000]


train_dataset = train_dataset.reshape(
    train_dataset.shape[0], 28, 28, 1)


import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

# Initialization
image_size = 28
truncatedN = keras.initializers.\
    TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
sgd = keras.optimizers.SGD(lr=0.03, decay=1e-5, momentum=0, nesterov=True)

# Initialising the ANN
classifier = Sequential()
classifier.add(Convolution2D(64, (3, 3), input_shape=(image_size, image_size, 1),
                             activation = 'relu', kernel_initializer=truncatedN))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Convolution2D(32, (3, 3),
                             activation = 'relu', kernel_initializer=truncatedN))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Flatten())
classifier.add(Dense(units=128, activation='relu',
                     kernel_initializer=truncatedN))
classifier.add(Dense(units=10, activation='softmax',
                     kernel_initializer=truncatedN))

# Compiling the ANN
classifier.compile(optimizer = sgd, loss = 'categorical_crossentropy',
                   metrics = ['accuracy'])


datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=10,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=False)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(train_dataset)

# fits the model on batches with real-time data augmentation:
classifier.fit_generator(datagen.flow(
    train_dataset, train_labels, batch_size=64),
    steps_per_epoch=len(train_dataset) / 64, epochs=50, verbose=2)

valid_dataset = valid_dataset.reshape(
    valid_dataset.shape[0], image_size, image_size, 1)
test_dataset = test_dataset.reshape(
    test_dataset.shape[0], image_size, image_size, 1)

# Predicting the Test set results
y_pred_valid = classifier.predict(valid_dataset)
y_pred_test = classifier.predict(test_dataset)

sum = np.sum(np.argmax(y_pred_valid[i]) == np.argmax(valid_labels[i])
             for i in range(0, valid_labels.shape[0]))
acc_valid = sum*1.0/5000
sum = np.sum(np.argmax(y_pred_test[i]) == np.argmax(test_labels[i])
             for i in range(0, test_labels.shape[0]))
acc_test = sum*1.0/5000

test_data = test_dataset[1000].reshape(1, image_size, image_size, 1)
classifier.predict(test_data)
plt.imshow(test_data.reshape(image_size, image_size), cmap='gray')
plt.show()

from scipy import misc
num = misc.imread('download.png')
num = num[:,:,0].reshape(image_size, image_size)
plt.imshow(num, cmap='gray')
plt.show()

num = num.reshape(1, image_size, image_size, 1)
num = 1.0 * (num - 127.5) / 255.0
classifier.predict(num)


# Save a keras model
from keras.models import load_model
classifier.save_weights('model.hdf5')
with open('model.json', 'w') as f:
    f.write(classifier.to_json())