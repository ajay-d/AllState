import numpy as np
np.random.seed(666)

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

train = pd.read_csv('train_recode_factor.csv.gz', compression="gzip")
test = pd.read_csv('test_recode_factor.csv.gz', compression="gzip")
train.shape
test.shape

train_id = train['id'].values
test_id = test['id'].values

y_train = train['loss'].ravel()

shift = 200
y_train_log = np.log(y_train + shift)

X = train.drop(['id', 'loss'], axis=1)
X_test = test.drop(['id'], axis=1)

nfolds = 5
kf = KFold(nfolds, shuffle=True)

etr_1 = ExtraTreesRegressor(n_estimators=5, max_depth=4, n_jobs=4)
etr_2 = ExtraTreesRegressor(n_estimators=10, max_depth=5, n_jobs=4)
etr_3 = ExtraTreesRegressor(n_estimators=5, max_depth=4, n_jobs=4)
etr_4 = ExtraTreesRegressor(n_estimators=10, max_depth=5, n_jobs=4)

boost_etr_1 = AdaBoostRegressor(etr_1, n_estimators = 25, random_state=333)
boost_etr_2 = AdaBoostRegressor(etr_2, n_estimators = 25, random_state=333)
boost_etr_3 = AdaBoostRegressor(etr_1, n_estimators = 50, random_state=444, learning_rate=.5)
boost_etr_4 = AdaBoostRegressor(etr_2, n_estimators = 50, random_state=444, learning_rate=.5)
boost_etr_5 = AdaBoostRegressor(etr_3, n_estimators = 50, random_state=555, learning_rate=.5)
boost_etr_6 = AdaBoostRegressor(etr_4, n_estimators = 50, random_state=555, learning_rate=.5)

pred_final = np.zeros(test.shape[0])
models = [boost_etr_1, boost_etr_2, boost_etr_3, boost_etr_4, boost_etr_5, boost_etr_6]

#To store stacked values
train_stack = pd.DataFrame(train_id, columns = ['id'])
test_stack = pd.DataFrame(test_id, columns = ['id'])

for i in np.arange(len(models))+1:
    train_stack["xtra_%d" % (i)] = np.zeros(train_stack.shape[0])
    test_stack["xtra_%d" % (i)] = np.zeros(test_stack.shape[0])

fold = 1
for train_index, test_index in kf.split(X):
    X_model, X_oos = X.values[train_index], X.values[test_index]
    y_model, y_oos = y_train_log[train_index], y_train_log[test_index]
    col = 1
    for i in models:
        i.fit(X_model, y_model)
        train_stack.ix[test_index, col] = i.predict(X_oos)
        test_stack.ix[:, col] += i.predict(X_test.values)
        pred_oos = i.predict(X_oos)
        print("Fold", fold, "model", col, "mae oos: ", np.mean(abs(np.exp(pred_oos)-200-np.exp(y_oos)-200)))
        col += 1
    fold += 1

for i in np.arange(len(models))+1:
    test_stack.ix[:, i] /= nfolds

train_stack.to_csv('train_stack.csv', index = False)
test_stack.to_csv('test_stack.csv', index = False)