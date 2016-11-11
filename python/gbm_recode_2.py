import numpy as np
np.random.seed(666)

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold

train = pd.read_csv('train_recode_factor2.csv.gz', compression="gzip")
test = pd.read_csv('test_recode_factor2.csv.gz', compression="gzip")
train.shape
test.shape

train_id = train['id'].values
test_id = test['id'].values

y_train = train['loss'].ravel()

shift = 200
y_train = np.log(y_train + shift)

X = train.drop(['id', 'loss'], axis=1)
X_test = test.drop(['id'], axis=1)

#res = xgb.cv(params, dtrain=xgb.DMatrix(X, label=y_train),
#num_boost_round=2000, nfold=5, stratified=False, 
#early_stopping_rounds=10,
#verbose_eval=1, show_stdv=True, metrics={'mae'})

params = {   
        'objective': 'reg:linear',
        'colsample_bytree': 0.3424,
        'min_child_weight': 3.65,
        'eval_metric': 'mae',
        'subsample': 0.88,
        'max_depth': 15,
        'nthread': 12,
        'silent': 1,
        'gamma': 2,
        'alpha': 2.37,
        'eta': .001
        }

nfolds = 5
kf = KFold(nfolds, shuffle=True)
pred_final = np.zeros(X_test.shape[0])
i = 0
mae_oos = 0
model = 1

##To store all CV results
df_results = pd.DataFrame(np.arange(0, nfolds, 1), columns=['fold'])
df_results['fold'] += 1
df_results['key'] = 1
df_results['mae'] = np.zeros(df_results.shape[0])

df_models = pd.DataFrame(np.arange(1, 4, 1), columns=['model'])
df_models['key'] = 1

df_results = pd.merge(df_models, df_results, on='key', how='inner', sort=True)
df_results.drop(['key'], axis=1, inplace=True)

xgtest = xgb.DMatrix(X_test.values)
for train_index, test_index in kf.split(X.values):
    X_model, X_oos = X.values[train_index], X.values[test_index]
    y_model, y_oos = y_train[train_index], y_train[test_index]
    dtrain = xgb.DMatrix(X_model, label=y_model)
    dtest = xgb.DMatrix(X_oos, label=y_oos)
    watchlist  = [(dtrain,'train'), (dtest,'test_oos')]
    evals_result = {}
    bst = xgb.train(params, dtrain, 100000, watchlist, 
                    early_stopping_rounds=300, evals_result=evals_result,
                    verbose_eval=1000)
    print("Best itr", bst.best_iteration)
    pred_oos = bst.predict(dtest, ntree_limit=bst.best_iteration)
    #df_results[df_results['model']==1].iloc[i, df_results.columns.get_loc('mae')] = np.mean(abs(np.exp(pred_oos)-200-np.exp(y_oos)-200))
    i += 1
    df_results.loc[(df_results['fold']==i) & (df_results['model']==model), 'mae'] = np.mean(abs(np.exp(pred_oos)-200-np.exp(y_oos)-200))
    pred_final += bst.predict(xgtest, ntree_limit=bst.best_iteration)
    mae_oos += np.mean(abs(np.exp(pred_oos)-200-np.exp(y_oos)-200))
    print("Fold", i, "mae oos: ", np.mean(abs(np.exp(pred_oos)-200-np.exp(y_oos)-200)))

pred_final
predictions = np.exp(pred_final/nfolds) - shift
##
print("Final MAE oos: ", mae_oos/nfolds)

df = pd.DataFrame(test_id, columns = ['id'])
df['gbm_bagged_1'] = predictions


params = {   
        'objective': 'reg:linear',
        'colsample_bytree': 0.49,
        'min_child_weight': 4.5,
        'eval_metric': 'mae',
        'subsample': 0.98,
        'max_depth': 13,
        'nthread': 12,
        'silent': 1,
        'gamma': 2.76,
        'alpha': 1.13,
        'eta': .001
        }

nfolds = 5
kf = KFold(nfolds, shuffle=True)
pred_final = np.zeros(X_test.shape[0])
i = 0
mae_oos = 0
model += 1

xgtest = xgb.DMatrix(X_test.values)
for train_index, test_index in kf.split(X.values):
    X_model, X_oos = X.values[train_index], X.values[test_index]
    y_model, y_oos = y_train[train_index], y_train[test_index]
    dtrain = xgb.DMatrix(X_model, label=y_model)
    dtest = xgb.DMatrix(X_oos, label=y_oos)
    watchlist  = [(dtrain,'train'), (dtest,'test_oos')]
    evals_result = {}
    bst = xgb.train(params, dtrain, 100000, watchlist, 
                    early_stopping_rounds=300, evals_result=evals_result,
                    verbose_eval=1000)
    print("Best itr", bst.best_iteration)
    #print(evals_result['test_oos']['mae'])
    pred_oos = bst.predict(dtest, ntree_limit=bst.best_iteration)
    i += 1
    df_results.loc[(df_results['fold']==i) & (df_results['model']==model), 'mae'] = np.mean(abs(np.exp(pred_oos)-200-np.exp(y_oos)-200))
    pred_final += bst.predict(xgtest, ntree_limit=bst.best_iteration)
    mae_oos += np.mean(abs(np.exp(pred_oos)-200-np.exp(y_oos)-200))
    print("Fold", i, "mae oos: ", np.mean(abs(np.exp(pred_oos)-200-np.exp(y_oos)-200)))

pred_final
predictions = np.exp(pred_final/nfolds) - shift
##
print("Final MAE oos: ", mae_oos/nfolds)

df['gbm_bagged_2'] = predictions

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
        'eta': .001
        }

nfolds = 5
kf = KFold(nfolds, shuffle=True)
pred_final = np.zeros(X_test.shape[0])
i = 0
mae_oos = 0
model += 1

xgtest = xgb.DMatrix(X_test.values)
for train_index, test_index in kf.split(X.values):
    X_model, X_oos = X.values[train_index], X.values[test_index]
    y_model, y_oos = y_train[train_index], y_train[test_index]
    dtrain = xgb.DMatrix(X_model, label=y_model)
    dtest = xgb.DMatrix(X_oos, label=y_oos)
    watchlist  = [(dtrain,'train'), (dtest,'test_oos')]
    evals_result = {}
    bst = xgb.train(params, dtrain, 100000, watchlist, 
                    early_stopping_rounds=300, evals_result=evals_result,
                    verbose_eval=1000)
    print("Best itr", bst.best_iteration)
    #print(evals_result['test_oos']['mae'])
    pred_oos = bst.predict(dtest, ntree_limit=bst.best_iteration)
    i += 1
    df_results.loc[(df_results['fold']==i) & (df_results['model']==model), 'mae'] = np.mean(abs(np.exp(pred_oos)-200-np.exp(y_oos)-200))
    pred_final += bst.predict(xgtest, ntree_limit=bst.best_iteration)
    mae_oos += np.mean(abs(np.exp(pred_oos)-200-np.exp(y_oos)-200))
    print("Fold", i, "mae oos: ", np.mean(abs(np.exp(pred_oos)-200-np.exp(y_oos)-200)))

pred_final
predictions = np.exp(pred_final/nfolds) - shift
##
print("Final MAE oos: ", mae_oos/nfolds)

df['gbm_bagged_3'] = predictions

df_results

df.to_csv('gbm_full2.csv', index = False)
