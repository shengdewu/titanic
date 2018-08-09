from sklearn import svm
import numpy as np

class csvm(object):
    def __init__(self):
        self.__train_kernel = svm.SVC(C=2.0,kernel='rbf',gamma=0.001)
        pass

    def train(self, train_data, test_data):
        '''
        :param train_data: (ndims*1,1)
        :param test_data:
        :return:
        '''
        #X, Y = self.__splite_data(train_data)

        self.__train_kernel.fit(train_data[1], train_data[0])

        if test_data:
            y_predict = self.__train_kernel.predict(test_data[1])

            predict_result = []
            if len(y_predict) == len(test_data[0]):
                for i in range(len(y_predict)):
                    predict_result.append((y_predict[i], test_data[0][i]))

            if predict_result:
                correct_cnt = sum(int(x == y) for x, y in predict_result)
                print("tatal %d correct %d %lf" %(len(y_predict), correct_cnt, correct_cnt/len(y_predict)))
            else:
                print("test data invalid")


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

