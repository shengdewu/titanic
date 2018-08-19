from data_preprocess import data_preprocess
from csvm import csvm
from load_data import load_data
from sklearn import svm
import numpy as np

if __name__ == '__main__':
    print(__name__)
    pre_process = data_preprocess()
    data = pre_process.process_data(path = './data/train.csv')

    # ld = load_data()
    # train_data, test_data = ld.load_train_data('./test_data/train.csv')
    #
    clf = csvm()
    #
    clf.train(train_data=data)
    #
    # precdict_data = pre_process.load_csv_data(path='./data/test.csv', test = True)
    #
    # print("load....")
    # print(clf.predict(precdict_data[0]))


