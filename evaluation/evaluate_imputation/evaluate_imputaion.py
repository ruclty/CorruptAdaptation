import pandas as pd
import numpy as np
from pandas.api.types import is_object_dtype
from math import sqrt


def imp_acc(miss_data, imp_data, ground_data):
    #  miss_data 是原来有缺失的数据，格式为dataframe
    #  imp_data 是填补好了的数据，格式为dataframe
    #  ground_data 是原来的真实数据，格式为dataframe
    #  上述三者的列名，索引应一致
    stat_na = miss_data.isnull().sum(axis=0)
    com_imp = imp_data[miss_data.isnull()]
    com_ground = ground_data[miss_data.isnull()]
    count_same = ground_data.eq(imp_data).astype(int).sum(axis=0)
    count_wrong = len(miss_data) - count_same
    count_right = stat_na - count_wrong
    attr_acc = stat_na.copy().astype(float)
    num_right = 0
    num_na = 0
    num_num_na = 0 # categorical
    nrms_num = 0
    for column in miss_data.columns:
        if is_object_dtype(miss_data[column]):
            if stat_na[column] == 0:
                attr_acc[column] = np.nan
            else:
                attr_acc[column] = count_right[column]/stat_na[column]
                num_right += count_right[column]
                num_na += stat_na[column]
        else:
            if stat_na[column] == 0:
                attr_acc[column] = np.nan
            else:
                error = (com_imp[column] - com_ground[column]).fillna(0)
                norm_error = sum(error*error)/(com_ground[column].var())
                attr_acc[column] = sqrt(norm_error/stat_na[column])
                nrms_num += norm_error
                num_num_na += stat_na[column]
    if num_na != 0:
        acc = num_right / num_na
    else:
        acc = np.nan
    if num_num_na != 0:
        nrms = nrms_num / num_num_na
    else:
        nrms = np.nan
    #  返回一个Series，对应每个类别型属性的填充准确率或者均方根误差，如果没有缺失就是NAN；
    #  返回一个整体填充准确率（categories），和一个整体均方根误差（numerical）
    return attr_acc, acc, nrms


if __name__ == "__main__":
    
    attr_acc, acc, nrms = imp_acc(miss_data, imp_data, ground_data)
    

            
