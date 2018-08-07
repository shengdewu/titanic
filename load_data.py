import os
import csv
import numpy as np
import random

class load_data(object):

    def load_train_data(self, path):
        file_path = os.path.abspath(path)
        fdata = csv.reader(open(file_path, 'r'))
        data = []
        for line in fdata:
            if 'label' == line[0]:
                continue
            data_y = int(line[0])
            data_x = line[1:]
            data_x = [int(i)/255 for i in data_x]
            data_x = np.array(data_x).reshape(len(data_x), 1)
            data.append((data_x, data_y))

        np.random.shuffle(data)
        test_data = data[: int(len(data) * 0.3)]
        training_data = data[int(len(data) * 0.3):]

        return training_data, test_data

    def laod_test_data(self, path):
        file_path = os.path.abspath(path)
        fdata = csv.reader(open(file_path, 'r'))
        test_data =[]
        for line in fdata:
            if 'pixel0' == line[0]:
                continue
            data = [int(i)/255 for i in line]
            data = np.array(data).reshape(len(data), 1)
            test_data.append(data)
        return test_data

    def save_predict(self, label, path):
        file_path = os.path.abspath(path)
        fdata = csv.writer(open(file_path, 'w'))
        id = 0
        for y in label:
            fdata.writerow([id, y])
            id +=1


