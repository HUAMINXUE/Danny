import pandas as pd
import numpy as np
from ini.ini import *
from enviorment.enviorment import Environment
from grach.grach import Grach
from data_serise.data_serise import Daily_Data_Serise
from analysis.analysis import Analysis
# from lstm.lstm import Lstm
if __name__ == '__main__':
    env = Environment()
    dds = Daily_Data_Serise()
    dds.read_local_file('index.h5')
    env.add_data(dds,Index_Data)

    # grach = Grach(env)
    # grach.run_grach(is_local=True)
    # lstm = Lstm(env)
    # lstm.build_net()
    # lstm.data_process()
    # lstm.run()
    analysis = Analysis(env)
    analysis.analysis()
    pass