import numpy as np
from keras import layers
from keras import regularizers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

import keras.backend as K
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import sys
from array import array

# io below
def readBytes(number, source):
    byte = source.read(number)
    return byte == b"", int.from_bytes(byte, byteorder='big')

def loadLabels(filename):
    number_of_items = 0
    labels = []

    with open(filename, "rb") as f:
        magic_number = readBytes(4, f)
        eof, number_of_items = readBytes(4, f)

        while True:
            eof, byte = readBytes(1, f);
            if eof:
                break
            labels.append(byte)

    return number_of_items, labels;

def loadImages(filename):
    number_of_items = 0
    images = []

    with open(filename, "rb") as f:
        magic_number = readBytes(4, f)
        _, number_of_items = readBytes(4, f)
        _, number_of_rows = readBytes(4, f)
        eof, number_of_columns = readBytes(4, f)

        flat = []
        data = array('B', f.read())
        for byte in data:
            flat.append(byte)

        images = np.array(flat).reshape(number_of_items, number_of_rows, number_of_columns)

        return number_of_items, images

def loadDataset(DEBUG = False):
    train_labels_no, train_labels = loadLabels("./dataset/train-labels-idx1-ubyte")
    test_labels_no, test_labels = loadLabels("./dataset/t10k-labels-idx1-ubyte")

    if DEBUG:
        print("Number of train labels: " + str(train_labels_no))
        print("Number of test labels: " + str(test_labels_no))

    train_images_no, train_images = loadImages("./dataset/train-images-idx3-ubyte")
    test_images_no, test_images = loadImages("./dataset/t10k-images-idx3-ubyte")

    if DEBUG:
        print("Number of train examples: " + str(train_images_no))
        print("Number of test examples: " + str(test_images_no))

    return train_labels, test_labels, train_images, test_images

# helpers below
def onehot(num, noe):
    cold = np.zeros(noe)
    cold[num] = 1
    return cold

def intFromOnehot(onehot):
    return np.where(onehot == 1)

def loadTrainingData():
     DEBUG = False

     train_labels, test_labels, train_images_orig, test_images_orig = loadDataset(DEBUG);

     if DEBUG:
         from matplotlib import pyplot as plt
         plt.imshow(train_images_orig[2137], interpolation='nearest')
         plt.show()

     # preprocess labels, migrate to onehot

     buff = []
     for label in train_labels:
         buff.append(onehot(label, 10)) # [1, 0, ..., 0] - 0; [0, 1, 0, ..., 0] - 1 ... and so on
     train_labels = np.array(buff)

     buff = []
     for label in test_labels:
         buff.append(onehot(label, 10))
     test_labels = np.array(buff)

     train_images = np.expand_dims(train_images_orig, axis=3);
     test_images = np.expand_dims(test_images_orig, axis=3);

     if DEBUG:
        print("train_images shape: " + str(train_images.shape))
        print("train_labels shape: " + str(train_labels.shape))

        print("test_images shape: " + str(test_images.shape))
        print("test_labels shape: " + str(test_labels.shape))

     # normalize the images
     train_images = train_images / 255.
     test_images = test_images / 255.

     return train_labels, test_labels, train_images, test_images, train_images_orig, test_images_orig

if __name__ == "__main__":
    #loading the dataset
    train_labels, test_labels, train_images, test_images, train_images_orig, test_images_orig = loadTrainingData();

    ## model fitting below ##
    X_input = Input((28, 28, 1))

    # Padding to the 32x32
    X = ZeroPadding2D((4, 4))(X_input)

    # CONV -> BN -> RELU
    X = Conv2D(6, (5, 5), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('tanh')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool_1', strides=2)(X)

    # CONV -> BN -> RELU
    X = Conv2D(16, (5, 5), strides = (1, 1), name = 'conv1')(X)
    X = BatchNormalization(axis = 3, name = 'bn1')(X)
    X = Dropout(0.4, name='drop_1')(X)

    X = Activation('tanh')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), strides=2, name='max_pool_2')(X)

    # FLATTEN
    X = Flatten()(X)
    X = Dropout(0.3, name="drop_2")(X)
    X = Dense(400, activation='tanh', name='fc_in')(X)
    X = Dense(120, activation='tanh', name='fc_mid')(X)
    X = Dense(84, activation='tanh', name='fc_out')(X)
    X = Dense(10, activation='softmax', name='fc_softmax')(X)

    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet-5')
    model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Fit the model
    model.fit(x=train_images, y=train_labels, epochs=5, batch_size=256)

    # Evaluate
    preds = model.evaluate(x = test_images, y = test_labels)

    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))
