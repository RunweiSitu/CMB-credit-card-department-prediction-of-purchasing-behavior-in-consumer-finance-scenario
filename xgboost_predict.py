# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

OFF_LINE = False


def xgb_model(train_set_x, train_set_y, test_set_x):
    # 模型参数
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'eta': 0.02,
              'max_depth': 6,
              'colsample_bytree': 0.9,
              'subsample': 0.7,
              'min_child_weight': 10,
              'silent': 1,
              }

    dtrain = xgb.DMatrix(train_set_x, label=train_set_y)
    dvali = xgb.DMatrix(test_set_x)
    watchlist = [(dtrain, 'train')]
    model = xgb.train(params, dtrain, num_boost_round=800, evals=watchlist)  # early_stopping_rounds=200
    predict = model.predict(dvali)

    # 保存特征重要性得分
    feature_score = model.get_fscore()
    feature_score = sorted(feature_score.items(), key=lambda x: x[1], reverse=True)
    fs = []
    for (key, value) in feature_score:
        fs.append("{0},{1}\n".format(key, value))
    with open('xgb_feature_score.csv', 'w') as f:
        f.writelines("feature,score\n")
        f.writelines(fs)

    return predict

def cross_validation(train_set, OFF_LINE):

    if OFF_LINE == True:
        train_x = train_set.drop(['USRID', 'FLAG'], axis=1).values
        train_y = train_set['FLAG'].values
        auc_list = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)
        for train_index, test_index in skf.split(train_x, train_y):
            print('Train: %s | test: %s' % (train_index, test_index))
            X_train, X_test = train_x[train_index], train_x[test_index]
            y_train, y_test = train_y[train_index], train_y[test_index]
            pred_value = xgb_model(X_train, y_train, X_test)
            print(pred_value)
            print(y_test)

            pred_value = np.array(pred_value)
            pred_value = [ele + 1 for ele in pred_value]

            y_test = np.array(y_test)
            y_test = [ele + 1 for ele in y_test]

            fpr, tpr, thresholds = roc_curve(y_test, pred_value, pos_label=2)
            auc = metrics.auc(fpr, tpr)
            print('auc value:', auc)
            auc_list.append(auc)
    print('validate result:', np.mean(auc_list))

def get_result(train_set,test_set):
    result_name = test_set[['USRID']]
    train_x = train_set.drop(['USRID', 'FLAG'], axis=1).values
    train_y = train_set['FLAG'].values
    test_x = test_set.drop(['USRID'], axis=1).values
    pred_result = xgb_model(train_x, train_y, test_x)
    result_name['RST'] = pred_result
    result_name.to_csv('xgb_submit.csv', index=None, sep='\t')

def main():
    all_train = pd.read_csv('all_train.csv', sep='\t')
    test_set = pd.read_csv('test_set.csv', sep='\t')
    cross_validation(all_train, True)
    get_result(all_train, test_set)

if __name__ == '__main__':
    main()