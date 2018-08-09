from parse_data import parse_data
from csvm import csvm
from load_data import load_data
from sklearn import svm
import numpy as np

if __name__ == '__main__':
    print(__name__)
    parse = parse_data()
    data = parse.load_csv_data(path = './data/train.csv')

    # ld = load_data()
    # train_data, test_data = ld.load_train_data('./test_data/train.csv')
    #
    clf = csvm()

    clf.train(train_data=data[0], test_data = data[1])

    precdict_data = parse.load_csv_data(path='./data/test.csv', test = True)

    print("load....")
    print(clf.predict(precdict_data[0]))


