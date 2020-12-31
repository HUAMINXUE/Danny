import pandas as pd
import os
from constant.constant import *
def change_df(df):
    return df[0:4]+'-'+df[4:6]+'-'+df[6:8]
data = pd.read_excel(os.path.join(DATA_BASE_PATH,'index.xlsx'))
data['trade_date']= list(map(change_df,data['trade_date'].values.astype(str)))
h = pd.HDFStore(os.path.join(DATA_BASE_PATH,'index.h5'),'w')
h['data'] = data
h.close()
op=1