from constant.constant import *
from ini.ini import *
import pandas as pd
import numpy as np
import statsmodels.tsa.api as smt
import statsmodels.api as sm
from arch import arch_model
import matplotlib.pyplot as plt
class Grach:
    def __init__(self,env):
        self.__env = env

    def get_env(self):
        return self.__env

    def set_env(self,env):
        self.__env = env

    def run_grach(
            self,
            index_name = 'SP500',
            is_local = True,
    ):
        if is_local:
            predict = pd.read_hdf(os.path.join(RESULTS,'train.h5'))
        else:
            index = self.get_env(
            ).query_data(Index_Data).get_data_serise(
            ).set_index(COM_DATE)
            index.index  = pd.to_datetime(index.index)
            ret = index/index.shift(1)-1
            ret = ret[
                ret.index>=pd.to_datetime('2000-01-01')
            ]
            ret = np.log(ret+1)
            trade_list = list(ret.index)
            trade_list.sort()
            begin_index = 256
            n = len(trade_list)-begin_index-100
            predict = pd.DataFrame()
            flag = False
            for i in range(n):
                print(i)
                ret_train = ret[
                    (ret.index<=trade_list[begin_index+i+100]) &
                    (ret.index>=trade_list[i+101])
                ]
                ret_train2 = ret[
                    (ret.index<=trade_list[begin_index+i+100]) &
                    (ret.index>=trade_list[i+100])
                ]
                ret_train3 = ret[
                    (ret.index <= trade_list[begin_index + i + 100]) &
                    (ret.index >= trade_list[i])
                ]
                predict_cell = ret[
                    (ret.index==trade_list[begin_index+i+100])
                ][[index_name]]
                am = arch_model(
                    ret_train[index_name].dropna(),
                    mean='ARX',
                    lags=0,
                    vol='egarch',
                    p=1,
                    o=0,
                    q=1,
                )
                model = am.fit()
                std = model.forecast(horizon=1)
                predict_cell['egarch']=np.sqrt(std.variance.iloc[-1].values[0])
                am = arch_model(
                    ret_train[index_name].dropna(),
                    mean='ARX',
                    lags=0,
                    vol='garch',
                    p=1,
                    o=0,
                    q=1,
                )
                model = am.fit()
                std = model.forecast(horizon=1)
                predict_cell['garch']=np.sqrt(std.variance.iloc[-1].values[0])
                u30 = ret_train3[index_name].values[-100:]
                vol302 = np.zeros(100)
                vol302[0] = ret_train3[index_name].values[-(100+begin_index):-100].std()
                for i in range(99):
                    vol302[i + 1] = np.sqrt(0.06 * u30[i] ** 2 + 0.94 * vol302[i] ** 2)
                predict_cell['garch'] =  vol302[-1]
                predict = pd.concat(
                    [predict,predict_cell],
                    axis = 0
                )
                if(flag):
                    predict.loc[trade_list[begin_index+i-1],'label'] = np.std(ret_train2[index_name].dropna())
                flag = True
            del predict[index_name]
            h = pd.HDFStore(os.path.join(RESULTS,'train.h5'),'w')
            h['data'] = predict
            h.close()
        self.get_env().add_data(predict,'train')