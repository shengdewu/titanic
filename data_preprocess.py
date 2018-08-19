import pandas as pd
import os
import numpy as np
import data_visual
import sklearn.preprocessing as sk_pre

def  length(x):
    if x is np.nan:
        return 0
    return len(x)


class data_preprocess(object):
    def __init__(self):
        print('init parse_data')


    def process_data(self, path, test = False):
        '''
        :param path:
        :return:
        '''
        abs_path = os.path.abspath(path)
        csv_data = pd.read_csv(abs_path, header=0, dtype={'Age': np.float, 'Fare':np.float})

        data = self.clear_feature(csv_data, test)

        return data

    def clear_feature(self, data_set, test = False):

        data_visual_tool = data_visual.data_visual()

        #data_visual_tool.draw_info(data_set)

        #data_visual_tool.draw_historgram(data_set, 'Fare')
        #data_visual_tool.draw_mutl_historgram(data_set=data_set, col='Survived', row='Pclass', attr='Age')
        #attr = ('Sex','Fare')
        #data_visual_tool.draw_point(data_set, 'Embarked', *attr)
        #data_visual_tool.draw_barplot(data_set, 'Embarked', 'Survived', *attr)

        data_set['Sex'] = data_set['Sex'].apply(lambda x: 1 if 'male' == x else 0)

        #创建新的特征
        data_set['Title'] = data_set['Name'].str.extract('([A-Za-z]+)\.', expand=False)
        #交叉表
        print(pd.crosstab(data_set['Title'], data_set['Sex']))

        data_set['Title'] = data_set['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', \
                                                     'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],
                                                    'Rare')

        data_set['Title'] = data_set['Title'].replace('Mlle', 'Miss')
        data_set['Title'] = data_set['Title'].replace('Ms', 'Miss')
        data_set['Title'] = data_set['Title'].replace('Mme', 'Mrs')

        #self.group_and_sort(data_set, 'Title')
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        data_set['Title'] = data_set['Title'].map(title_mapping)
        data_set['Title'] = data_set['Title'].fillna(0)


        #使用众数填充
        #data_set['Age'] = self.fill_with_mode(data_set, 'Age')
        guess_ages = np.zeros((2, 3))
        for i in range(0, 2):
            for j in range(0, 3):
                guess_df = data_set[(data_set['Sex'] == i) & (data_set['Pclass'] == j + 1)]['Age'].dropna()

                # age_mean = guess_df.mean()
                # age_std = guess_df.std()
                # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

                age_guess = guess_df.median()

                # Convert random age float to nearest .5 age
                guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

        for i in range(0, 2):
            for j in range(0, 3):
                data_set.loc[(data_set.Age.isnull()) & (data_set.Sex == i) & (data_set.Pclass == j + 1), 'Age'] = guess_ages[i, j]

        data_set['Age'] = data_set['Age'].astype(int)

        data_set['Fare'] = self.fill_with_mode(data_set, 'Fare')
        data_set['Embarked'] = self.fill_with_mode(data_set, 'Embarked')

        #归一化特征
        #data_set = self.group_and_sort(data_set, 'Age', True, 5)

        data_set.loc[data_set['Age'] <= 16, 'Age'] = 0
        data_set.loc[(data_set['Age'] > 16) & (data_set['Age'] <= 32), 'Age'] = 1
        data_set.loc[(data_set['Age'] > 32) & (data_set['Age'] <= 48), 'Age'] = 2
        data_set.loc[(data_set['Age'] > 48) & (data_set['Age'] <= 64), 'Age'] = 3
        data_set.loc[data_set['Age'] > 64, 'Age']

        data_set['FamilySize'] = data_set['SibSp'] + data_set['Parch'] + 1
        data_set = self.group_and_sort(data_set, 'FamilySize')

        data_set['IsAlone'] = 0
        data_set.loc[data_set['FamilySize']==1, 'IsAlone'] = 1

        #data_set = self.group_and_sort(data_set, 'Fare', True, 4)

        data_set.loc[data_set['Fare'] <= 17.91, 'Fare'] = 0
        data_set.loc[(data_set['Fare'] > 17.91) & (data_set['Fare'] <= 14.454), 'Fare'] = 1
        data_set.loc[(data_set['Fare'] > 14.454) & (data_set['Fare'] <= 31.0), 'Fare'] = 2
        data_set.loc[data_set['Fare'] > 31.0, 'Fare'] = 3


        data_set['Embarked'] = data_set['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

        data_set['Age*Class'] = data_set['Age'] * data_set['Pclass']

        data_set = data_set.drop(labels='Cabin', axis=1)
        data_set = data_set.drop(labels='FamilySize', axis=1)
        data_set = data_set.drop(labels='SibSp', axis=1)
        data_set = data_set.drop(labels='Parch', axis=1)
        data_set = data_set.drop(labels='Ticket', axis=1)
        data_set = data_set.drop(labels='Name', axis=1)
        data_set = data_set.drop(labels='PassengerId', axis=1)

        data = []
        if not test:
            y_train = data_set['Survived']
            x_train = data_set.drop(labels='Survived', axis=1)
            data.append(y_train)
            data.append(x_train)
        else:
            data.append(data_set)

        return data

    def fill_with_mode(self, data_set, attr):
        '''
        使用众数填充
        :param data_set:
        :param attr:
        :return:
        '''
        freq = data_set[attr].dropna().mode()[0]
        return data_set[attr].fillna(freq)

    def group_and_sort(self, data_set, attr, cut=False, cnt=None):
        '''
        分组并排序
        :param data_set:
        :param attr:
        :param cut:
        :return:
        '''
        attr_band = attr
        if cut:
            attr_band = attr + 'Band'
            '''
            pd.cut 根据值本身来划分 适用与连续值
            pd.qcut 根据值出现的频率 适用于离散的
            '''
            data_set[attr_band] = pd.cut(data_set[attr], cnt)

        result = data_set[[attr_band, 'Survived']].groupby([attr_band], as_index=False).mean().sort_values(by=attr_band, ascending=True)

        print(result)

        if cut:
            data_set = data_set.drop(labels=attr_band, axis=1)


        print(result)


        return data_set

