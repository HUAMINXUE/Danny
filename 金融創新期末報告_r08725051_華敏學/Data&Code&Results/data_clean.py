import pandas as pd
from constant.constant import *
if __name__ == '__main__':
    garch = pd.DataFrame()
    ewma = pd.DataFrame()
    egarch = pd.DataFrame()
    garch_lstm = pd.DataFrame()
    lstm = pd.DataFrame()
    count = 0
    for file in ['D_J','GG','SP500','PX_LAST','Nada']:
        data1 = pd.read_csv(os.path.join(RESULTS,file+'_train.csv'))
        if(count==0):
            garch['trade_date'] = data1['trade_date']
            ewma['trade_date'] = data1['trade_date']
            egarch['trade_date'] = data1['trade_date']
        garch[file] = data1['garch']
        ewma[file] = data1['ewma']
        egarch[file] = data1['egarch']
        del data1['label']
        data2 = pd.read_csv(os.path.join(RESULTS,file+'_single_predict.csv')).rename(columns = {
            'vol':'lstm',
            'Unnamed: 0':'trade_date'
        })
        if(count==0):
            lstm['trade_date'] = data2['trade_date']
        lstm[file] = data2['lstm']
        data3 = pd.read_csv(os.path.join(RESULTS, file + '_predict.csv')).rename(columns = {
            'vol':'garch_lstm',
            'Unnamed: 0':'trade_date'
        })
        if(count==0):
            garch_lstm['trade_date'] = data3['trade_date']
        garch_lstm[file] = data3['garch_lstm']
        count +=1
    h = pd.HDFStore(os.path.join(RESULTS,'garch_vol.h5'))
    h['data'] = garch
    h.close()
    h = pd.HDFStore(os.path.join(RESULTS,'ewma_vol.h5'))
    h['data'] = ewma
    h.close()
    h = pd.HDFStore(os.path.join(RESULTS,'egarch_vol.h5'))
    h['data'] = egarch
    h.close()
    h = pd.HDFStore(os.path.join(RESULTS,'lstm_vol.h5'))
    h['data'] = lstm
    h.close()
    h = pd.HDFStore(os.path.join(RESULTS,'garch_lstm_vol.h5'))
    h['data'] = garch_lstm
    h.close()