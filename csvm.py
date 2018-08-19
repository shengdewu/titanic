from sklearn import svm
import numpy as np

class csvm(object):
    def __init__(self):
        self.__train_kernel = svm.SVC(C=2.0,kernel='rbf',gamma=0.001)
        pass

    def train(self, train_data):
        '''
        :param train_data: (ndims*1,1)
        :param test_data:
        :return:
        '''
        #X, Y = self.__splite_data(train_data)

        self.__train_kernel.fit(train_data[1], train_data[0])

        acc_svc = round(self.__train_kernel.score(train_data[1], train_data[0]) * 100, 2)
        print("correct %f" % acc_svc)

        return


    def predict(self, input):
        '''
        :param input: array_like nsample * nfeature
        :return:
        '''
        return self.__train_kernel.predict(input)

    def __splite_data(self, data):
        if not data:
            return
        nfeatures = len(data[0][0])
        nsamples = len(data)

        X = np.zeros(shape=(nsamples, nfeatures), dtype=np.float)
        Y = []
        for i in range(nsamples):
            val = data[i][0]
            X[i] = val.T
            Y.append(data[i][1])

        return X, Y

