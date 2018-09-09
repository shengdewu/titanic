import pandas as pd

class chi_square(object):
    '''
    卡方检验:只支持单一因素校验
    '''
    def checkout(self, data_set, condition):
        '''
        :param data_set:
        :param condition: the first one is must label
        :return:
        '''
        data_ci = data_set.loc[:, condition]
        data_group = data_ci.groupby(condition[0])
        #统计各个属性在每个标签的频率
        freq_0 = (data_group[condition].count()).drop(condition[0], axis=1)
        #构造列联表
        row_sum = freq_0.sum(axis=1)
        freq = freq_0.assign(row_sum=row_sum)
        col_sum = freq.sum(axis=0)
        freq = freq.append(col_sum, ignore_index=True)
        #计算卡方的理论值
        total = col_sum[-1]
        freq_e = []
        for row in row_sum:
            freq_t = []
            for col in col_sum[0:-1]:
                freq_t.append(col * row / total)
            freq_e.append(freq_t)

        freq_e = pd.DataFrame(freq_e, columns=condition[1:])

        #Calculate the chi-square check value
        freq = freq_0.sub(freq_e)
        freq = freq.div(freq_e)
        chi_val = (freq.sum()).sum()
        #计算自由度
        freedom = freq_e.shape
        freedom = (freedom[0]-1) * (freedom[1]-1)

        return freedom, chi_val
