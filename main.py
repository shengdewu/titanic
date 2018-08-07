from csv_data_parse import csv_data_parse
from csvm import csvm
from load_data import load_data
from sklearn import svm
import numpy as np

if __name__ == '__main__':
    print(__name__)

    # X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1], [40, 60], [50, 70]])
    # y = np.array([1, 1, 2, 2, 3, 3])
    #
    # clf = svm.SVC(C=2.0,kernel='rbf',gamma=0.001)
    # clf.fit(X, y)
    #
    # print(clf.predict([[-0.8, -1]]))
    #
    # print(clf.predict([[1, 2]]))
    #
    # print(clf.predict([[-1, -1], [-2, -1], [1, 1], [2, 1], [40, 60], [50, 70]]))


    # csv_parse = csv_data_parse()
    # train_data = csv_parse.load_csv_data(path = './data/train.csv')
    # print(train_data['PassengerId'])
    ld = load_data()
    train_data, test_data = ld.load_train_data('./test_data/train.csv')

    clf = csvm()

    clf.train(train_data=train_data, test_data=test_data)


