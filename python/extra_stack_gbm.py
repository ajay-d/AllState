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

train_stack = pd.read_csv('train_stack.csv')
test_stack = pd.read_csv('test_stack.csv')

train_id = train['id'].values
test_id = test['id'].values

y_train = train['loss'].ravel()

shift = 200
y_train = np.log(y_train + shift)

##Add stacking values
test_all = pd.merge(test, test_stack, on='id', how='inner', sort=False)
train_all = pd.merge(train, train_stack, on='id', how='inner', sort=False)

X = train_all.drop(['id', 'loss'], axis=1)
X_test = test_all.drop(['id'], axis=1)

params = {   
        'objective': 'reg:linear',
        'colsample_bytree': 0.5,
        'min_child_weight': 1,
        'eval_metric': 'mae',
        'subsample': 0.8,
        'max_depth': 12,
        'nthread': 12,
        'silent': 1,
        'gamma': 1,
        'alpha': 1,
        'eta': .01
        }

nfolds = 5
kf = KFold(nfolds, shuffle=True)

#res = xgb.cv(params, dtrain=xgb.DMatrix(X, label=y_train),
#num_boost_round=2000, nfold=5, stratified=False, 
#early_stopping_rounds=10,
#verbose_eval=1, show_stdv=True, metrics={'mae'})

pred_final = np.zeros(X_test.shape[0])
i = 0
mae_oos = 0

xgtest = xgb.DMatrix(X_test.values)
for train_index, test_index in kf.split(X.values):
    X_model, X_oos = X.values[train_index], X.values[test_index]
    y_model, y_oos = y_train[train_index], y_train[test_index]
    dtrain = xgb.DMatrix(X_model, label=y_model)
    dtest = xgb.DMatrix(X_oos, label=y_oos)
    watchlist  = [(dtrain,'train'), (dtest,'test_oos')]
    evals_result = {}
    bst = xgb.train(params, dtrain, 5000, watchlist, 
                    early_stopping_rounds=50, evals_result=evals_result,
                    verbose_eval=100)
    print("Best itr", bst.best_iteration)
    #print(evals_result['test_oos']['mae'])
    i += 1
    pred_oos = bst.predict(dtest, ntree_limit=bst.best_iteration)
    pred_final += bst.predict(xgtest, ntree_limit=bst.best_iteration)
    mae_oos += np.mean(abs(np.exp(pred_oos)-200-np.exp(y_oos)-200))
    print("Fold", i, "mae oos: ", np.mean(abs(np.exp(pred_oos)-200-np.exp(y_oos)-200)))

pred_final
predictions = np.exp(pred_final/nfolds) - shift
##1207.74 without stack
##
print("Final MAE oos: ", mae_oos/nfolds)

df = pd.DataFrame(test_id, columns = ['id'])
df['gbm_bagged_1'] = predictions


params = {   
        'objective': 'reg:linear',
        'colsample_bytree': 0.5,
        'colsample_bylevel': .9,
        'eval_metric': 'mae',
        'subsample': 0.9,
        'max_depth': 9,
        'nthread': 12,
        'silent': 1,
        'gamma': .1,
        'alpha': .1,
        'eta': .005
        }

nfolds = 6
kf = KFold(nfolds, shuffle=True)
pred_final = np.zeros(X_test.shape[0])
i = 0
mae_oos = 0

xgtest = xgb.DMatrix(X_test.values)
for train_index, test_index in kf.split(X.values):
    X_model, X_oos = X.values[train_index], X.values[test_index]
    y_model, y_oos = y_train[train_index], y_train[test_index]
    dtrain = xgb.DMatrix(X_model, label=y_model)
    dtest = xgb.DMatrix(X_oos, label=y_oos)
    watchlist  = [(dtrain,'train'), (dtest,'test_oos')]
    evals_result = {}
    bst = xgb.train(params, dtrain, 5000, watchlist, 
                    early_stopping_rounds=50, evals_result=evals_result,
                    verbose_eval=100)
    print("Best itr", bst.best_iteration)
    #print(evals_result['test_oos']['mae'])
    i += 1
    pred_oos = bst.predict(dtest, ntree_limit=bst.best_iteration)
    pred_final += bst.predict(xgtest, ntree_limit=bst.best_iteration)
    mae_oos += np.mean(abs(np.exp(pred_oos)-200-np.exp(y_oos)-200))
    print("Fold", i, "mae oos: ", np.mean(abs(np.exp(pred_oos)-200-np.exp(y_oos)-200)))

pred_final
predictions = np.exp(pred_final/nfolds) - shift
##1208 without stack
##
print("Final MAE oos: ", mae_oos/nfolds)

df['gbm_bagged_2'] = predictions

df.to_csv('gbm_full_stack.csv', index = False)
