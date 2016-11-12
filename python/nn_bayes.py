import numpy as np
np.random.seed(666)

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from bayes_opt import BayesianOptimization
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU, PReLU

#Baysian hyperparameter optimization [https://github.com/fmfn/BayesianOptimization]

#https://github.com/fchollet/keras/issues/3857
tf.python.control_flow_ops = tf

train = pd.read_csv('train_recode_factor.csv.gz', compression="gzip")
test = pd.read_csv('test_recode_factor.csv.gz', compression="gzip")
train.shape
test.shape

all_data = pd.concat([train.drop(['loss'], axis=1), test], ignore_index=True)
all_data.index

df_normal = pd.DataFrame(all_data, columns = ['id'])
for f in all_data.columns:
    if 'cat' in f:
        d = pd.get_dummies(all_data[f])
        frames = [df_normal, d]
        df_normal = pd.concat(frames, axis=1)

f_num = [f for f in all_data.columns if 'cont' in f]
s = (all_data[f_num] - all_data[f_num].mean()) / all_data[f_num].std()
frames = [df_normal, s]
df_normal = pd.concat(frames, axis=1)

##Mean normalize data
train_id = train['id'].values
test_id = test['id'].values

y_train = train['loss'].ravel()

#y_train = np.log(y_train)

test_normal = pd.merge(test['id'].to_frame(), df_normal, on='id', how='inner', sort=False)
train_normal = pd.merge(train['id'].to_frame(), df_normal, on='id', how='inner', sort=False)

train_normal.drop(['id', 'loss'], axis=1, inplace=True)
test_normal.drop(['id', 'loss'], axis=1, inplace=True)

data_test = test_normal
data_train = train_normal

optim = {}
optim[1] = 'RMSprop'
optim[2] = 'Adadelta'
optim[3] = 'Adam'
optim[4] = 'Adamax'
optim[5] = 'Nadam'

init = {}
init[1] = 'lecun_uniform'
init[2] = 'glorot_normal'
init[3] = 'glorot_uniform'
init[4] = 'he_normal'
init[5] = 'he_uniform'

def nn_evaluate(layer_1, layer_2, dropout_1, dropout_2, optimizer, initialization):
  
    model = Sequential()
    model.add(Dense(round(layer_1), input_dim = data_train.shape[1], init = init[round(initialization)]))
    model.add(PReLU())
    model.add(Dropout(dropout_1))
    model.add(Dense(round(layer_2), input_dim = data_train.shape[1], init = init[round(initialization)]))
    model.add(PReLU())
    model.add(Dropout(dropout_2))
    model.add(Dense(1, init = init[round(initialization)]))
    model.compile(optimizer=optim[round(optimizer)], loss='mae')
    
    mae_result = 0
    X = data_train.values
    y = y_train
    for train_index, test_index in kf.split(X):
        X_model, X_oos = X[train_index], X[test_index]
        y_model, y_oos = y[train_index], y[test_index]
        pred_oos = np.zeros(y_oos.shape[0])
        model.fit(X_model, y_model, batch_size=500, nb_epoch=nepochs, shuffle=True, verbose=0)
        pred_oos = model.predict(X_oos)[:,0]
        mae_result += np.mean(abs(pred_oos-y_oos))
        
    return -mae_result / nfolds

nfolds = 2
nepochs = 10
kf = KFold(nfolds, shuffle=True)

num_iter = 10
init_points = 10

nnBO = BayesianOptimization(nn_evaluate, {'layer_1': (50, 1000),
                                          'layer_2': (50, 1000),
                                          'dropout_1': (0.1, 0.9),
                                          'dropout_2': (0.1, 0.9),
                                          'optimizer': (1, 5),
                                          'initialization': (1, 5)
                                          })

nnBO.maximize(init_points=init_points, n_iter=num_iter)
