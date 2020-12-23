import numpy as np
import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD
import os
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def lenet():
    model = Sequential()
    # first conv and pool
    model.add(Conv2D(input_shape=(28, 28, 1), kernel_size=(5, 5), filters=20, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
    # second conv and pool
    model.add(Conv2D(kernel_size=(5, 5), filters=50, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))

    model.add(Flatten())
    model.add(Dense(500, activation='relu'))  # fc1
    model.add(Dense(10, activation='softmax'))  # fc2
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    # data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    X_train = X_train / 255
    X_test = X_test / 255
    y_train = np_utils.to_categorical(y_train, num_classes=10)  # label onehot化
    y_test = np_utils.to_categorical(y_test, num_classes=10)

    # train and test
    model = lenet()
    print('Training')
    history = model.fit(X_train, y_train, epochs=1, batch_size=32, validation_split=0.2)
    print('\nTesting')
    text_loss, text_accuracy = model.evaluate(X_test, y_test)

    print('\ntest loss: ', text_loss)
    print('\ntest accuracy: ', text_accuracy)

    # confusion

    print_confusion_result(X_train, X_test, np.argmax(y_train, axis=1), np.argmax(y_test, axis=1), model)

    # save model
    model.save('./model/lenet.h5')  