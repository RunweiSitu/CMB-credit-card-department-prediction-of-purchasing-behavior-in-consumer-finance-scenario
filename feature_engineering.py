# -*- coding: utf-8 -*-
import pandas as pd
from datetime import datetime
import numpy as np
from pandas.core.frame import DataFrame

"""
构造特征,保存到log_pre.csv
day_of_week_0_sum到day_of_week_6_sum表示用户周一到周日的模块点击次数
将一天划分成多个时间段后，time_of_day_0_sum到time_of_day_6_sum 分别统计用户在各个时间段的模块点击次数
TCH_TYP_0_sum 和 TCH_TYP_2_sum 表示用户对不同事件类型的模块点击次数
click_count表示用户点击的模块总数
EVT_LBL_set_len表示用户点击了多少个独特的模块（跟click_count相比，去掉了重复的模块)
"""

def log_pre():
    train_agg = pd.read_csv('train/train_agg.csv', sep='\t')
    train_flg = pd.read_csv('train/train_flg.csv', sep='\t')
    train_log = pd.read_csv('train/train_log.csv', sep='\t')
    test_agg = pd.read_csv('test/test_agg.csv', sep='\t')
    test_log = pd.read_csv('test/test_log.csv', sep='\t')

    log = pd.concat([train_log, test_log], copy=False)
    log['day_of_week'] = log['OCC_TIM'].map(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").weekday())
    log['day'] = log['OCC_TIM'].map(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").day)
    ## sleep: 0-6 , go to work:7-9, work:10-12, sleep:13-14, work 15-17,dinner 18-20, rest 21-23
    time_of_day = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]
    log['time_of_day'] = log['OCC_TIM'].map(lambda x: time_of_day[datetime.strptime(x, "%Y-%m-%d %H:%M:%S").hour])

    dw = pd.get_dummies(log["day_of_week"], prefix="day_of_week")
    td = pd.get_dummies(log["time_of_day"], prefix="time_of_day")
    tch = pd.get_dummies(log["TCH_TYP"], prefix="TCH_TYP")
    log = pd.concat([log, dw, td, tch], axis=1)

    log['click_count'] = 1
    c0 = log.groupby(['USRID'], as_index=False)['click_count'].agg({'click_count': np.sum})
    c1 = log.groupby(['USRID'], as_index=False)['EVT_LBL'].agg({'EVT_LBL_set_len': lambda x: len(set(x))})

    log.drop(['EVT_LBL', 'OCC_TIM', 'TCH_TYP', 'day_of_week', 'day', 'time_of_day'], axis=1, inplace=True)

    d0 = log.groupby(['USRID'], as_index=False)['day_of_week_0'].agg({'day_of_week_0_sum': np.sum})
    d1 = log.groupby(['USRID'], as_index=False)['day_of_week_1'].agg({'day_of_week_1_sum': np.sum})
    d2 = log.groupby(['USRID'], as_index=False)['day_of_week_2'].agg({'day_of_week_2_sum': np.sum})
    d3 = log.groupby(['USRID'], as_index=False)['day_of_week_3'].agg({'day_of_week_3_sum': np.sum})
    d4 = log.groupby(['USRID'], as_index=False)['day_of_week_4'].agg({'day_of_week_4_sum': np.sum})
    d5 = log.groupby(['USRID'], as_index=False)['day_of_week_5'].agg({'day_of_week_5_sum': np.sum})
    d6 = log.groupby(['USRID'], as_index=False)['day_of_week_6'].agg({'day_of_week_6_sum': np.sum})

    t0 = log.groupby(['USRID'], as_index=False)['time_of_day_0'].agg({'time_of_day_0_sum': np.sum})
    t1 = log.groupby(['USRID'], as_index=False)['time_of_day_1'].agg({'time_of_day_1_sum': np.sum})
    t2 = log.groupby(['USRID'], as_index=False)['time_of_day_2'].agg({'time_of_day_2_sum': np.sum})
    t3 = log.groupby(['USRID'], as_index=False)['time_of_day_3'].agg({'time_of_day_3_sum': np.sum})
    t4 = log.groupby(['USRID'], as_index=False)['time_of_day_4'].agg({'time_of_day_4_sum': np.sum})
    t5 = log.groupby(['USRID'], as_index=False)['time_of_day_5'].agg({'time_of_day_5_sum': np.sum})
    t6 = log.groupby(['USRID'], as_index=False)['time_of_day_6'].agg({'time_of_day_6_sum': np.sum})

    p0 = log.groupby(['USRID'], as_index=False)['TCH_TYP_0'].agg({'TCH_TYP_0_sum': np.sum})
    p2 = log.groupby(['USRID'], as_index=False)['TCH_TYP_2'].agg({'TCH_TYP_2_sum': np.sum})

    log_pre = pd.merge(d0, d1, on=['USRID'], how='left', copy=False)
    log_pre = pd.merge(log_pre, d2, on=['USRID'], how='left', copy=False)
    log_pre = pd.merge(log_pre, d3, on=['USRID'], how='left', copy=False)
    log_pre = pd.merge(log_pre, d4, on=['USRID'], how='left', copy=False)
    log_pre = pd.merge(log_pre, d5, on=['USRID'], how='left', copy=False)
    log_pre = pd.merge(log_pre, d6, on=['USRID'], how='left', copy=False)
    log_pre = pd.merge(log_pre, t0, on=['USRID'], how='left', copy=False)
    log_pre = pd.merge(log_pre, t1, on=['USRID'], how='left', copy=False)
    log_pre = pd.merge(log_pre, t2, on=['USRID'], how='left', copy=False)
    log_pre = pd.merge(log_pre, t3, on=['USRID'], how='left', copy=False)
    log_pre = pd.merge(log_pre, t4, on=['USRID'], how='left', copy=False)
    log_pre = pd.merge(log_pre, t5, on=['USRID'], how='left', copy=False)
    log_pre = pd.merge(log_pre, t6, on=['USRID'], how='left', copy=False)
    log_pre = pd.merge(log_pre, p0, on=['USRID'], how='left', copy=False)
    log_pre = pd.merge(log_pre, p2, on=['USRID'], how='left', copy=False)
    log_pre = pd.merge(log_pre, c0, on=['USRID'], how='left', copy=False)
    log_pre = pd.merge(log_pre, c1, on=['USRID'], how='left', copy=False)

    log_pre.to_csv('log_pre.csv', sep='\t', index=False)


"""
构造特征,保存到log_EVT_square.csv
将用户的点击模块划分成三个级别EVT_LBL_1，EVT_LBL_2，EVT_LBL_3，统计用户在一个级别内对不同模块的点击次数
如特征EVT_1_num_12表示用户对第一个级别的第12个模块的点击次数
特征EVT_2_num_21表示用户对第二个级别的第21个模块的点击次数，依此类推。。。。。。
"""

def log_EVT_square():
    train_log = pd.read_csv('train/train_log.csv', sep='\t')
    test_log = pd.read_csv('test/test_log.csv', sep='\t')
    log = pd.concat([train_log, test_log], copy=False)

    log['EVT_LBL_1'] = log['EVT_LBL'].apply(lambda x: x.split('-')[0])
    log['EVT_LBL_2'] = log['EVT_LBL'].apply(lambda x: x.split('-')[1])
    log['EVT_LBL_3'] = log['EVT_LBL'].apply(lambda x: x.split('-')[2])

    full_EVT_1 = list(set(log['EVT_LBL_1']))
    full_EVT_2 = list(set(log['EVT_LBL_2']))
    full_EVT_3 = list(set(log['EVT_LBL_3']))

    def EVT_LBL_1_TONUM(s):
        return full_EVT_1.index(s)

    def EVT_LBL_2_TONUM(s):
        return full_EVT_2.index(s)

    def EVT_LBL_3_TONUM(s):
        return full_EVT_3.index(s)

    log['EVT_LBL_1'] = log.EVT_LBL_1.apply(EVT_LBL_1_TONUM)
    log['EVT_LBL_2'] = log.EVT_LBL_2.apply(EVT_LBL_2_TONUM)
    log['EVT_LBL_3'] = log.EVT_LBL_3.apply(EVT_LBL_3_TONUM)

    Users_set = DataFrame(list(set(log['USRID'])))
    Users_set.columns = ['USRID']

    ###统计每个用户各个level的不同模块的点击次数，i表示full_EVT的第i个元素
    t = log[['USRID', 'EVT_LBL_1']][log.EVT_LBL_1 == 0]
    t['EVT_1_num_0'] = 1
    t = t.groupby(['USRID', 'EVT_LBL_1']).agg('sum').reset_index()
    t.drop('EVT_LBL_1', axis=1, inplace=True)
    log_EVT_1 = pd.merge(Users_set, t, on='USRID', how='left')

    for i in range(1, len(full_EVT_1)):
        t = log[['USRID', 'EVT_LBL_1']][log.EVT_LBL_1 == i]
        t['EVT_1_num_' + str(i)] = 1
        t = t.groupby(['USRID', 'EVT_LBL_1']).agg('sum').reset_index()
        t.drop('EVT_LBL_1', axis=1, inplace=True)
        log_EVT_1 = pd.merge(log_EVT_1, t, on='USRID', how='left')

    for i in range(len(full_EVT_1)):
        log_EVT_1['EVT_1_num_' + str(i)][log_EVT_1['EVT_1_num_' + str(i)].isnull()] = 0

    t = log[['USRID', 'EVT_LBL_2']][log.EVT_LBL_2 == 0]
    t['EVT_2_num_0'] = 1
    t = t.groupby(['USRID', 'EVT_LBL_2']).agg('sum').reset_index()
    t.drop('EVT_LBL_2', axis=1, inplace=True)
    log_EVT_2 = pd.merge(Users_set, t, on='USRID', how='left')

    for i in range(1, len(full_EVT_2)):
        t = log[['USRID', 'EVT_LBL_2']][log.EVT_LBL_2 == i]
        t['EVT_2_num_' + str(i)] = 1
        t = t.groupby(['USRID', 'EVT_LBL_2']).agg('sum').reset_index()
        t.drop('EVT_LBL_2', axis=1, inplace=True)
        log_EVT_2 = pd.merge(log_EVT_2, t, on='USRID', how='left')

    for i in range(len(full_EVT_2)):
        log_EVT_2['EVT_2_num_' + str(i)][log_EVT_2['EVT_2_num_' + str(i)].isnull()] = 0

    t = log[['USRID', 'EVT_LBL_3']][log.EVT_LBL_3 == 0]
    t['EVT_3_num_0'] = 1
    t = t.groupby(['USRID', 'EVT_LBL_3']).agg('sum').reset_index()
    t.drop('EVT_LBL_3', axis=1, inplace=True)
    log_EVT_3 = pd.merge(Users_set, t, on='USRID', how='left')

    for i in range(1, len(full_EVT_3)):
        t = log[['USRID', 'EVT_LBL_3']][log.EVT_LBL_3 == i]
        t['EVT_3_num_' + str(i)] = 1
        t = t.groupby(['USRID', 'EVT_LBL_3']).agg('sum').reset_index()
        t.drop('EVT_LBL_3', axis=1, inplace=True)
        log_EVT_3 = pd.merge(log_EVT_3, t, on='USRID', how='left')

    for i in range(len(full_EVT_3)):
        log_EVT_3['EVT_3_num_' + str(i)][log_EVT_3['EVT_3_num_' + str(i)].isnull()] = 0

    log_EVT = pd.merge(log_EVT_1, log_EVT_2, on='USRID', how='left')
    log_EVT = pd.merge(log_EVT, log_EVT_3, on='USRID', how='left')

    ## 对用户点击模块的次数取平方
    user = log_EVT.pop('USRID')
    log_EVT_square = np.square(log_EVT)
    user = DataFrame(user)
    log_EVT_square = pd.concat([user, log_EVT_square], axis=1)
    log_EVT_square.to_csv('log_EVT_square.csv', sep='\t', index=False)

"""
构造新的用户时间特征，与之前的log_pre.csv，log_EVT_square.csv合并，
形成最终的训练集all_train.csv，最终的测试集test_set.csv
"""

def log_tabel(data):
    data['day'] = data['OCC_TIM'].map(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").day)
    data['hour'] = data['OCC_TIM'].map(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").hour)
    data['min'] = data['OCC_TIM'].map(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").minute)
    EVT_LBL_len = data.groupby(by=['USRID'], as_index=False)['EVT_LBL'].agg({'EVT_LBL_len': len})
    # 小时均值特征
    t0 = data.groupby('USRID')['hour'].mean().reset_index()
    t0.columns = ['USRID', 'user_mean_hour']
    # 小时var方差特征
    t1 = data.groupby('USRID')['hour'].var().reset_index()
    t1.columns = ['USRID', 'user_var_hour']
    # 天均值特征
    t2 = data.groupby('USRID')['day'].mean().reset_index()
    t2.columns = ['USRID', 'user_mean_day']
    # 天方差特征
    t3 = data.groupby('USRID')['day'].var().reset_index()
    t3.columns = ['USRID', 'user_var_day']
    # 小时min,max,时间差特征
    t4 = data.groupby('USRID')['hour'].min().reset_index()
    t4.columns = ['USRID', 'user_min_hour']
    t5 = data.groupby('USRID')['hour'].max().reset_index()
    t5.columns = ['USRID', 'user_max_hour']
    diff = t5['user_max_hour'] - t4['user_min_hour']
    user = t4['USRID']
    t6 = pd.concat([user, diff], axis=1)
    t6.columns = ['USRID', 'user_diff_hour']
    # 天min,max,时间差特征
    t7 = data.groupby('USRID')['day'].min().reset_index()
    t7.columns = ['USRID', 'user_min_day']
    t8 = data.groupby('USRID')['day'].max().reset_index()
    t8.columns = ['USRID', 'user_max_day']
    diff2 = t8['user_max_day'] - t7['user_min_day']
    user2 = t7['USRID']
    t9 = pd.concat([user2, diff2], axis=1)
    t9.columns = ['USRID', 'user_diff_day']

    return EVT_LBL_len, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9

def Merge():
    train_agg = pd.read_csv('train/train_agg.csv', sep='\t')
    train_flg = pd.read_csv('train/train_flg.csv', sep='\t')
    train_log = pd.read_csv('train/train_log.csv', sep='\t')

    all_train = pd.merge(train_flg, train_agg, on=['USRID'], how='left')
    EVT_LBL_len, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9 = log_tabel(train_log)
    all_train = pd.merge(all_train, EVT_LBL_len, on=['USRID'], how='left')
    all_train = pd.merge(all_train, t0, on=['USRID'], how='left')
    all_train = pd.merge(all_train, t1, on=['USRID'], how='left')
    all_train = pd.merge(all_train, t2, on=['USRID'], how='left')
    all_train = pd.merge(all_train, t3, on=['USRID'], how='left')
    all_train = pd.merge(all_train, t4, on=['USRID'], how='left')
    all_train = pd.merge(all_train, t5, on=['USRID'], how='left')
    all_train = pd.merge(all_train, t6, on=['USRID'], how='left')
    all_train = pd.merge(all_train, t7, on=['USRID'], how='left')
    all_train = pd.merge(all_train, t8, on=['USRID'], how='left')
    all_train = pd.merge(all_train, t9, on=['USRID'], how='left')

    log_pre = pd.read_csv('log_pre.csv', sep='\t')
    all_train = pd.merge(all_train, log_pre, on=['USRID'], how='left', copy=False)
    log_EVT_square = pd.read_csv('log_EVT_square.csv', sep='\t')
    all_train = pd.merge(all_train, log_EVT_square, on=['USRID'], how='left', copy=False)
    all_train.fillna(0, inplace=True)
    all_train.to_csv('all_train.csv', sep='\t', index=False)

    test_agg = pd.read_csv('test/test_agg.csv', sep='\t')
    test_log = pd.read_csv('test/test_log.csv', sep='\t')

    EVT_LBL_len, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9 = log_tabel(test_log)
    test_set = pd.merge(test_agg, EVT_LBL_len, on=['USRID'], how='left')
    test_set = pd.merge(test_set, t0, on=['USRID'], how='left')
    test_set = pd.merge(test_set, t1, on=['USRID'], how='left')
    test_set = pd.merge(test_set, t2, on=['USRID'], how='left')
    test_set = pd.merge(test_set, t3, on=['USRID'], how='left')
    test_set = pd.merge(test_set, t4, on=['USRID'], how='left')
    test_set = pd.merge(test_set, t5, on=['USRID'], how='left')
    test_set = pd.merge(test_set, t6, on=['USRID'], how='left')
    test_set = pd.merge(test_set, t7, on=['USRID'], how='left')
    test_set = pd.merge(test_set, t8, on=['USRID'], how='left')
    test_set = pd.merge(test_set, t9, on=['USRID'], how='left')

    test_set = pd.merge(test_set, log_pre, on=['USRID'], how='left', copy=False)
    test_set = pd.merge(test_set, log_EVT_square, on=['USRID'], how='left', copy=False)
    test_set.fillna(0, inplace=True)
    test_set.to_csv('test_set.csv', sep='\t', index=False)

def main():
    log_pre()
    log_EVT_square()
    Merge()

if __name__ == '__main__':
    main()
