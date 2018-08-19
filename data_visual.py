import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class data_visual(object):

    def plot_map(self, data_set):
        '''
        热力图 协方差
        :param data_set:
        :return:
        '''
        colormap = plt.cm.RdBu
        plt.figure(figsize=(14, 12))
        plt.title('Pearson Correlation of Features', y=1.05, size=15)
        sns.heatmap(data_set.astype(float).corr(), linewidths=0.1, vmax=1.0,
                    square=True, cmap=colormap, linecolor='white', annot=True)

        g = sns.pairplot(data_set[[u'Survived', u'Pclass', u'Sex', u'Age', u'SibSp', u'Parch', u'Fare', u'Cabin', u'Embarked',]],
                         hue='Survived', palette='seismic', size=1.2, diag_kind='kde',
                         diag_kws=dict(shade=True), plot_kws=dict(s=10))
        g.set(xticklabels=[])
        return

    def plot_gradit(self, data_set):
        features = data_set.columns
        features_name = list(features) + 'bias'

        plt.figure(figsize=(20,6))
        plt.ylabel('gradient of loss')
        plt.xlabel('per feature paramter $w_{d}$')
        plt.title('one step gradients')
        plt.xticks(rotation=45)



        x = data_set[features].values

        print(x)
        return
    def draw_info(self, data_set):
        '''
        统计方差，平均值，极值
        :param data_set:
        :return:
        '''
        data_set.info()
        print('_'*40)
        print(data_set.describe())
        print('_'*40)
        print(data_set.describe(include=['O']))
        return

    def draw(self, data_set):
        '''
        四位数图
        :param data_set:
        :return:
        '''
        sns.set_style("whitegrid")
        # data_set = data_set.drop(labels='Fare', axis=1)
        # data_set = data_set.drop(labels='Age', axis=1)
        sns.boxplot(data=data_set)

        return

    def draw_historgram(self, data_set, attr):
        '''
        historgram visualizations
        :param data_set:
        :param attr:
        :return:
        '''
        gf = sns.FacetGrid(data_set, col='Survived')
        gf.map(plt.hist, attr, bins=30)
        print('visualizations')

        return

    def draw_mutl_historgram(self, data_set, col, row, attr):
        '''
        mutl historgram visualizations
        :param data_set:
        :param attr:
        :return:
        '''
        gf = sns.FacetGrid(data_set, col=col, row=row, height=2.2, aspect=1.6)
        gf.map(plt.hist, attr, bins=30, alpha=0.5)
        gf.add_legend()
        print('visualizations')

        return

    def draw_point(self, data_set, row, *attr):
        '''
        折线图
        :param data_set:
        :param row:
        :param attr:
        :return:
        '''
        gf = sns.FacetGrid(data_set, row=row, size=2.2, aspect=1.6)
        gf.map(sns.pointplot, *attr, palette='deep')
        gf.add_legend()
        print('visualizations')
        return

    def draw_barplot(self, data_set, row, col, *attr):
        '''
        条形图
        :param data_set:
        :param row:
        :param col:
        :param attr:
        :return:
        '''
        gf = sns.FacetGrid(data_set, row=row, col=col, height=2.2, aspect=1.6)
        gf.map(sns.barplot, *attr, alpha=0.5, ci=None)
        gf.add_legend()
        print('visualizations')
        return