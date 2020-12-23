import pandas as pd
import numpy as np
from pyts.image import GASF
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras import backend as K
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Activation, MaxPool2D


# load data and process to the dict like pickle shows
def load_csv(path='./data/eurusd_2010_2017_1T_rulebase.csv'):
    # get X and Y
    df = pd.read_csv(path)
    cols = ['low', 'high', 'close', 'open', 'EveningStar', 'ShootingStar', 'BearishHarami', 'None']
    temp = df.loc[:, cols]
    data_temp = temp[(df[cols[-4]] + df[cols[-3]] + df[cols[-2]]) == 1]
    index = data_temp.index.values

    X1 = temp.loc[:, cols[0]].values
    X2 = temp.loc[:, cols[1]].values
    X3 = temp.loc[:, cols[2]].values
    X4 = temp.loc[:, cols[3]].values

    # use GASF transform
    image_size = 10
    gasf = GASF(image_size)
    print("the  data  processing's iter  start")
    for num, i in enumerate(index):
        n = (num + 1) % 1000
        if n == 0:
            print("the  data  processing's iter is ", num + 1)
        s = i - 9
        e = i + 1
        X_gasf1 = gasf.fit_transform(X1[s:e].reshape(1, -1))
        X_gasf2 = gasf.fit_transform(X2[s:e].reshape(1, -1))
        X_gasf3 = gasf.fit_transform(X3[s:e].reshape(1, -1))
        X_gasf4 = gasf.fit_transform(X4[s:e].reshape(1, -1))

        if num == 0:
            A1 = np.stack([X_gasf1, X_gasf2, X_gasf3, X_gasf4]).reshape(1, 4, 10, 10)
        else:
            A2 = np.stack([X_gasf1, X_gasf2, X_gasf3, X_gasf4]).reshape(1, 4, 10, 10)
            A1 = np.concatenate([A1, A2])
    print("the  data  processing's iter  over")

    # get data
    X = A1.reshape(A1.shape[0], 10, 10, 4)
    Y = data_temp.loc[:, cols[-4:-1]].values
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=0)
    data = {}
    data['train_x'] = train_x
    data['train_y'] = train_y
    data['test_x'] = test_x
    data['test_y'] = test_y

    return data


# model
def get_cnn_model(params):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(10, 10, 4)))
    model.add(Conv2D(filters=48, kernel_size=(5, 5), padding='valid', activation='relu'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    return model


# train
def train_model(params, data):
    model = get_cnn_model(params)
    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=['accuracy'])
    hist = model.fit(x=data['train_x'], y=data['train_y'],
                     batch_size=params['batch_size'], epochs=params['epochs'], verbose=2)
    return (model, hist)


# show
def print_result(data, model):
    # get train & test pred-labels
    train_pred = model.predict_classes(data['train_x'])
    test_pred = model.predict_classes(data['test_x'])
    # get train & test true-labels
    train_label = np.argmax(data['train_y'], axis=1)
    test_label = np.argmax(data['test_y'], axis=1)
    # confusion matrix
    train_result_cm = confusion_matrix(train_label, train_pred, labels=range(3))
    test_result_cm = confusion_matrix(test_label, test_pred, labels=range(3))
    print(train_result_cm, '\n' * 2, test_result_cm)

PARAMS = {}
PARAMS['classes'] = 3
PARAMS['lr'] = 0.01
PARAMS['epochs'] = 10
PARAMS['batch_size'] = 64
PARAMS['optimizer'] = optimizers.SGD(lr=PARAMS['lr'])


# # load data & keras model
data = load_csv()
# train cnn model
model, hist = train_model(PARAMS, data)
# train & test result
scores = model.evaluate(data['test_x'], data['test_y'], verbose=0)
print('CNN test accuracy:', scores[1])
print_result(data, model)
