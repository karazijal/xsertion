from keras.datasets import cifar10, cifar100
from keras.utils import np_utils
import keras.backend as K
import numpy as np
import PIL.Image as im

def get_cifar(p, append_test, use_c10):
    # The raw data, shuffled and split between training and test sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data() if use_c10 else cifar100.load_data()
    num_classes = 10 if use_c10 else 100

    if K.image_dim_ordering() == 'tf': # back to th ordering
        X_train = X_train.transpose(0, 3, 1, 2)
        X_test = X_test.transpose(0, 3, 1, 2)

    # convert from RBG to YUV
    for i in range(X_train.shape[0]):
        img = im.fromarray(np.transpose(X_train[i]))
        yuv = img.convert('YCbCr')
        X_train[i] = np.transpose(np.array(yuv))

    for i in range(X_test.shape[0]):
        img = im.fromarray(np.transpose(X_test[i]))
        yuv = img.convert('YCbCr')
        X_test[i] = np.transpose(np.array(yuv))

    if K.image_dim_ordering() == 'tf': # back to tf ordering from th
        X_train = X_train.transpose(0, 2, 3, 1)
        X_test = X_test.transpose(0, 2, 3, 1)

    # Compute how much to retain per class
    cnts = np.full(num_classes, ( X_train.shape[0] // num_classes) * p)

    rem = []
    X_val = []
    y_val = []

    for i in range(0,  X_train.shape[0]):
        cur_cls = y_train[i]
        if cnts[cur_cls] > 0:
            cnts[cur_cls] -= 1
        else:
            rem.append(i)
            X_val.append(X_train[i])
            y_val.append(cur_cls)

    # if append_test:
    #     X_test = np.append(X_test, gather_x, axis=0)
    #     y_test = np.append(y_test, gather_y, axis=0)
    #
    # Remove the computed indices
    X_train = np.delete(X_train, rem, 0)
    y_train = np.delete(y_train, rem, 0)
    # X_train_rgb = np.copy(X_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    # print(X_train.shape, X_val.shape, X_test.shape)
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, num_classes)
    Y_val = np_utils.to_categorical(y_val, num_classes)
    Y_test = np_utils.to_categorical(y_test, num_classes)
    X_train = X_train.astype('float64')
    X_test = X_test.astype('float64')
    X_val = X_val.astype('float64')

    return num_classes, X_train, Y_train, X_val, Y_val, X_test, Y_test


def val_split(val_p, X_train, Y_train):
    num_classes = Y_train[0].shape[0]
    cnts = np.full(num_classes, (X_train.shape[0] // num_classes) * val_p)
    X_t = []
    Y_t = []
    X_v = []
    Y_v = []
    for i in range(0, X_train.shape[0]):
        cur_cls = np.argmax(Y_train[i])
        if cnts[cur_cls] > 0:
            cnts[cur_cls] -= 1
            X_v.append(X_train[i])
            Y_v.append(Y_train[i])
        else:
            X_t.append(X_train[i])
            Y_t.append(Y_train[i])
    return np.array(X_t), np.array(Y_t), np.array(X_v), np.array(Y_v)


