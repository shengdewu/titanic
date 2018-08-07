import pandas as pd
import os
import numpy as np

class csv_data_parse(object):
    def __init__(self):
        print('init csv_data_parse')


    def load_csv_data(self, path):
        '''
        :param path:
        :return:
        '''
        abs_path = os.path.abspath(path)
        data = pd.read_csv(abs_path, header=0, dtype={'Age': np.float})
        return data