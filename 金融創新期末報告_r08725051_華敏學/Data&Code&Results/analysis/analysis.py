import pandas as pd
import numpy as np
import os
from constant.constant import *
from ini.ini import *
from sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller,q_stat,acf
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
class Analysis:
    def __init__(self,env):
        self.__env = env

    def get_env(self):
        return self.__env

    def set_env(self,env):
        self.__env = env


    def __analysis_index(self):
        index = self.get_env().query_data(Index_Data).get_data_serise()
        index_name = list(index.columns)
        index_name.remove(COM_DATE)
        index[index_name] = index[index_name].pct_change()/100
        index[index_name] = np.log(index[index_name]+1)
        index = index.set_index(COM_DATE)
        index.index = pd.to_datetime(index.index)
        res = pd.DataFrame(columns = ['mean','std','skew','kurt','jarque-Bera','adf','lm'])
        for index_name_ in index_name:
            fig, ax = plt.subplots()
            ax.plot(index[index_name_].dropna(), label=index_name_)
            ax.set_xlabel('时间')
            ax.set_ylabel('收益率的对数')
            ax.set_title(index_name_+'收益率图')
            ax.legend()
            plt.savefig(os.path.join(RESULTS, index_name_+'.png'))
            plt.close()
            fig, ax = plt.subplots()
            ax.hist(index[index_name_].dropna(),bins =25)
            ax.set_xlabel('收益率范围')
            ax.set_ylabel('收益率的对数')
            ax.set_title(index_name_+'收益率图')
            plt.savefig(os.path.join(RESULTS, index_name_+'bar.png'))
            plt.close()
            res.loc[index_name_] = [
                np.nanmean(index[index_name_].dropna()),
                np.nanstd(index[index_name_].dropna()),
                index[index_name_].dropna().skew(),
                index[index_name_].dropna().kurt(),
                stats.jarque_bera(index[index_name_].dropna())[0],
                adfuller(index[index_name_].dropna())[4]['5%'],
                q_stat(acf(index[index_name_].dropna())[1:13],len(index[index_name_].dropna()))[1][-1]
            ]
        res.to_csv(os.path.join(RESULTS,'index_info.csv'))


    def analysis(self):
        def hmse(x,y):
            return np.nanmean((1-x/(y+0.000000001))**2)

        def hmae(x,y):
            return np.nanmean(np.abs((1-x/(y+0.000000001))))
        #self.__analysis_index()
        for index_name in ['D_J','GG','PX_LAST','Nada','SP500']:
            print(index_name)
            if not os.path.exists(os.path.join(RESULTS,index_name)):
                os.makedirs(os.path.join(RESULTS,index_name))
            data = pd.read_csv(os.path.join(RESULTS,index_name+'_predict.csv')).rename(columns = {'Unnamed: 0':'trade_date'})
            data2 = pd.read_csv(os.path.join(RESULTS,index_name+'_single_predict.csv')).rename(columns = {'Unnamed: 0':'trade_date'})
            # data = pd.read_csv(os.path.join(RESULTS, 'train.csv'), index_col=False)
            market = pd.read_csv(os.path.join(RESULTS,index_name+'_train.csv'))
            data = pd.merge(
                data.rename(columns = {'vol':'garch_lstm'}),
                market,
                on = ['trade_date'],
                how = 'left'
            ).dropna()
            data = pd.merge(
                data2.rename(columns={'vol': 'lstm'}),
                data,
                on=['trade_date'],
                how='left'
            ).dropna()
            for col in ['ewma','garch','egarch','lstm','garch_lstm']:
                data[col] = data[col].fillna(method='ffill')
                data.loc[data[col]>0.04,col] = np.nan
                data.loc[data[col] <= 0, col] = np.nan
                data[col + '_std'] = data[col].shift(1).rolling(30, min_periods=5).std().fillna(method='ffill') / np.sqrt(30)
                data[col + '_std'] = data[col + '_std'].fillna(data[col + '_std'].dropna().values[0])
                data[col + '_mean'] = data[col].shift(1).rolling(30, min_periods=5).mean()
                data[col + '_mean'] = data[col + '_mean'].fillna(method='ffill').fillna(data[col].dropna().values[0])
                data[col + '_up'] = data[col + '_mean'] + 3 * data[col + '_std']
                data[col + '_down'] = data[col + '_mean'] - 3 * data[col + '_std']

                data.loc[data[col] > data[col + '_up'], col] = np.nan

                data.loc[data[col] <= data[col + '_down'], col] = np.nan
                data[col] = data[col].replace({np.inf:np.nan})
                data[col] = data[col].replace({-np.inf:np.nan})
                data[col] = data[col].fillna(method='ffill')
            data = data.set_index(COM_DATE)
            data = data.dropna()
            data.index = pd.to_datetime(data.index)
            res = pd.DataFrame(columns = ['garch_lstm','lstm','ewma','garch','egarch'])
            res.loc['mse'] = [
                mean_squared_error(data['label'],data['garch_lstm']),
                mean_squared_error(data['label'], data['lstm']),
                mean_squared_error(data['label'], data['ewma']),
                mean_squared_error(data['label'], data['garch']),
                mean_squared_error(data['label'], data['egarch'])
            ]
            res.loc['mae'] = [
                mean_absolute_error(data['label'],data['garch_lstm']),
                mean_absolute_error(data['label'], data['lstm']),
                mean_absolute_error(data['label'], data['ewma']),
                mean_absolute_error(data['label'], data['garch']),
                mean_absolute_error(data['label'], data['egarch'])
            ]
            res.loc['hmse'] = [
                hmse(data['label'],data['garch_lstm']),
                hmse(data['label'], data['lstm']),
                hmse(data['label'], data['ewma']),
                hmse(data['label'], data['garch']),
                hmse(data['label'], data['egarch'])
            ]
            res.loc['hmae'] = [
                hmae(data['label'],data['garch_lstm']),
                hmae(data['label'],data['lstm']),
                hmae(data['label'], data['ewma']),
                hmae(data['label'], data['garch']),
                hmae(data['label'], data['egarch'])
            ]
            for model in ['ewma','garch','egarch','garch_lstm','lstm']:
                if model!= 'label':
                    fig, ax = plt.subplots()
                    ax.plot(data[model], label=model)
                    ax.plot(data['label'], label='label')
                    ax.set_xlabel('时间')
                    ax.set_ylabel('波动率')
                    ax.set_title( '波动率图')
                    ax.legend()
                    plt.savefig(os.path.join(RESULTS,index_name, model+'_vol.png'))
                    plt.close()
            fig, ax = plt.subplots()
            for model in ['label','ewma','garch','egarch','garch_lstm','lstm']:
                ax.plot(data[model], label=model)
            ax.set_xlabel('时间')
            ax.set_ylabel('波动率')
            ax.set_title( '波动率图')
            ax.legend()
            plt.savefig(os.path.join(RESULTS,index_name, 'total.png'))
            plt.close()
            res.to_csv(os.path.join(RESULTS,index_name,'results.csv'))
        op=1