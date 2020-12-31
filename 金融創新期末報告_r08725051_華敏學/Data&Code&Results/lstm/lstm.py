# import keras as ks
import pandas as pd
import numpy as np
import os
from constant.constant import *
import pickle
class LSTM:
    def __init__(self,env):
        self.__env = env
        self.model = None

    def set_env(self,env):
        self.__env = env

    def get_env(self):
        return self.__env

    def data_process(self,is_local = True,time_zoom = 1):
        if is_local:
            data = pd.read_hdf(os.path.join(RESULTS,'train_clean.h5'))
        else:
            data = self.get_env().query_data('train')
            data['ewma']=pd.Series.ewm(data["label"], span=10).mean().shift(1)
            feature = list(data.columns)
            feature.remove('label')
            feature_list = []
            for log in [0,1,2,3]:
                for feature_ in feature:
                    feature_list.append(feature_ + '_log'+str(log))
                    data[feature_ + '_log'+str(log)] = data[feature_].shift(log)
            data = data.dropna()
        train = {
        }
        trade_list = list(data.index)
        trade_list.sort()
        begin_index = 752
        n = len(trade_list)-begin_index-time_zoom
        if n%time_zoom==0:
            m = int(n/time_zoom)
        else:
            m = int(n/time_zoom)+1
        for i in range(m):
            begin = trade_list[i*time_zoom]
            end = trade_list[begin_index+i*time_zoom-1]
            pred_begin = trade_list[begin_index+i*time_zoom]
            pred_end = trade_list[begin_index+i*time_zoom+time_zoom-1]
            train[pred_begin] = {
                'train':[],
                'label':[],
                'predict':[],
                'time':[]
            }
            train_use = data[
                (data.index>=begin) &
                (data.index<=end)
            ]
            predict_use = data[
                (data.index>=pred_begin) &
                (data.index<=pred_end)
            ]
            train[pred_begin]['train'] = train_use[feature_list].values.reshape((len(train_use),4,3))
            train[pred_begin]['label'] = list(train_use['label'].values)
            train[pred_begin]['predict'] = predict_use[feature_list].values.reshape((len(predict_use), 4, 3))
            train[pred_begin]['time'] = list(predict_use.index)
        with open(os.path.join(RESULTS,'train.pkl'),'wb') as f:
            pickle.dump(train,f)

        self.get_env().add_data(data,'train_clean')
        pass

    def build_net(self):
        # model = ks.models.Sequential()
        # model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
        # model.add(ks.layers.Dense(1024))
        # model.add(ks.layers.Dense(1))
        # model.compile(loss='mse', optimizer='adam')
        # self.model = model
        pass

    def run(self):
        train = self.get_env().query_data('train_clean')
        trade_list = list(train.index)
        trade_list.sort()
        begin_index = 252
        n = len(trade_list)-begin_index
        feature = list(train.columns)
        feature.remove('label')
        predict = pd.DataFrame()
        for i in range(n):
            begin_date = trade_list[i]
            end_date = trade_list[i+begin_index-1]
            trade_date = trade_list[i+begin_index]
            train_use= train[
                (train.index>=begin_date) &
                (train.index<=end_date)
            ]
            predict_cell = train[
                train.index==trade_date
            ]
            self.model.fit(train_use[feature],train_use["label"])
            predict_cell['y_hat'] = self.model.predict(predict[feature])
            predict = pd.concat(
                [predict,predict_cell],
                axis = 0
            )
        pass