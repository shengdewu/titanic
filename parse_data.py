import pandas as pd
import os
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

def  length(x):
    if x is np.nan:
        return 0
    return len(x)


class parse_data(object):
    def __init__(self):
        print('init parse_data')


    def load_csv_data(self, path, test = False):
        '''
        :param path:
        :return:
        '''
        abs_path = os.path.abspath(path)
        csv_data = pd.read_csv(abs_path, header=0, dtype={'Age': np.float, 'Fare':np.float})

        data = self.process_feature(csv_data, test)

        data_set = []
        if not test:
            size = len(data[0])
            test_len = int(size * 0.3)
            test_data = []
            train_data = []

            test_data.append(data[0][0:test_len])
            train_data.append(data[0][test_len:])

            test_data.append(data[1][0:test_len])
            train_data.append(data[1][test_len:])

            data_set.append(train_data)
            data_set.append(test_data)
        else:
            data_set = data

        return data_set

    def process_feature(self, data_set, test = False):

        #csv_data = csv_data.drop(labels='PassengerId',axis=1)
        data_set['Sex'] = data_set['Sex'].apply(lambda x: 1 if 'male' == x else 0)
        data_set['Cabin'] = data_set["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
        data_set['Embarked'] = data_set['Embarked'].fillna('S')
        data_set['Embarked'] = data_set['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
        ages = data_set['Age']
        age_avg = ages.sum()/ages.size
        data_set['Age'] = data_set['Age'].fillna(age_avg)

        fare = data_set['Fare']
        fare_avg = fare.sum()/fare.size
        data_set['Fare'] = data_set['Fare'].fillna(fare_avg)

        data_set = data_set.drop(labels='Ticket', axis=1)
        data_set = data_set.drop(labels='Name', axis=1)
        data_set = data_set.drop(labels='PassengerId', axis=1)

        data = []
        if not test:
            survived = data_set['Survived']
            data_set = data_set.drop(labels='Survived', axis=1)
            label = np.array(survived)
            data.append(label)

        data.append(np.array(data_set))

        # colormap = plt.cm.RdBu
        # plt.figure(figsize=(14, 12))
        # plt.title('Pearson Correlation of Features', y=1.05, size=15)
        # sns.heatmap(data_set.astype(float).corr(), linewidths=0.1, vmax=1.0,
        #             square=True, cmap=colormap, linecolor='white', annot=True)

        # g = sns.pairplot(data_set[[u'Survived', u'Pclass', u'Sex', u'Age', u'SibSp', u'Parch', u'Fare', u'Cabin', u'Embarked',]],
        #                  hue='Survived', palette='seismic', size=1.2, diag_kind='kde',
        #                  diag_kws=dict(shade=True), plot_kws=dict(s=10))
        # g.set(xticklabels=[])

        return data
