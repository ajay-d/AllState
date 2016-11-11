import numpy as np
np.random.seed(666)

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from bayes_opt import BayesianOptimization

#Baysian hyperparameter optimization [https://github.com/fmfn/BayesianOptimization]

train = pd.read_csv('train_recode_factor.csv.gz', compression="gzip")
test = pd.read_csv('test_recode_factor.csv.gz', compression="gzip")
train.shape
test.shape

train_id = train['id'].values
test_id = test['id'].values

y_train = train['loss'].ravel()

shift = 200
y_train = np.log(y_train + shift)

X = train.drop(['id', 'loss'], axis=1)
X_test = test.drop(['id'], axis=1)

def xgb_evaluate(min_child_weight, colsample_bytree, max_depth, subsample, gamma, alpha):
    
    params['min_child_weight'] = int(min_child_weight)
    params['cosample_bytree'] = max(min(colsample_bytree, 1), 0)
    params['max_depth'] = int(max_depth)
    params['subsample'] = max(min(subsample, 1), 0)
    params['gamma'] = max(gamma, 0)
    params['alpha'] = max(alpha, 0)
    cv_result = xgb.cv(params, dtrain, num_boost_round=num_rounds, nfold=5,
                       callbacks=[xgb.callback.early_stop(50)])
    return -cv_result['test-mae-mean'].values[-1]


    
num_rounds = 1000
num_iter = 10
init_points = 10

params = {
        'eta': 0.1,
        'silent': 1,
        'eval_metric': 'mae',
        'verbose_eval': True
    }
    
dtrain = xgb.DMatrix(X.values, label=y_train)
xgbBO = BayesianOptimization(xgb_evaluate, {'min_child_weight': (1, 5),
                                            'colsample_bytree': (0.3, 0.6),
                                            'max_depth': (10, 15),
                                            'subsample': (0.7, 1),
                                            'gamma': (0, 3),
                                            'alpha': (0, 3),
                                            })


xgbBO.maximize(init_points=init_points, n_iter=num_iter)

#alpha: 2.37
#colsample_bytree: .3424
#gamma: 2.05
#max_depth: 15
#min_child_weight: 3.65
#subsample: .88

#alpha: 1.13
#colsample_bytree: .49
#gamma: 2.76
#max_depth: 13
#min_child_weight: 4.5
#subsample: .98

print(xgbBO.res['max'])
