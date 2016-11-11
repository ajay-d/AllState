import numpy as np
np.random.seed(666)

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU

#https://github.com/fchollet/keras/issues/3857
tf.python.control_flow_ops = tf

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
#y_train = np.log(y_train + shift)

test_stack = pd.merge(test, test_stack, on='id', how='inner', sort=False)
train_stack = pd.merge(train, train_stack, on='id', how='inner', sort=False)

all_data = pd.concat([train_stack.drop(['loss'], axis=1), test_stack], ignore_index=True)
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

##Normalize stacked values
f_num = [f for f in all_data.columns if 'xtra' in f]
s = (all_data[f_num] - all_data[f_num].mean()) / all_data[f_num].std()
frames = [df_normal, s]
df_normal = pd.concat(frames, axis=1)

##Add stacking values
##Mean normalize data
test_normal = pd.merge(test['id'].to_frame(), df_normal, on='id', how='inner', sort=False)
train_normal = pd.merge(train['id'].to_frame(), df_normal, on='id', how='inner', sort=False)

train_normal.drop(['id', 'loss'], axis=1, inplace=True)
test_normal.drop(['id', 'loss'], axis=1, inplace=True)

data_test = test_normal
data_train = train_normal

nbags = 5
nfolds = 5
nepochs = 10

kf = KFold(nfolds, shuffle=True)

def bag_model_cv(X, y, nn_model):
    i = 0
    pred_final = np.zeros(data_test.shape[0])
    print(pred_final)
    for train_index, test_index in kf.split(X):
        X_model, X_oos = X[train_index], X[test_index]
        y_model, y_oos = y[train_index], y[test_index]
        #sample_index = np.arange(X_model.shape[0])
        pred_oos = np.zeros(y_oos.shape[0])
        for j in range(nbags):
            #np.random.shuffle(sample_index)
            #x_bag = X_model[sample_index]
            #y_bag = y_model[sample_index]
            nn_model.fit(X_model, y_model, batch_size=500, nb_epoch=nepochs, shuffle=True)
            pred_final += model.predict(data_test.values, batch_size=1000)[:,0]
            pred_oos += model.predict(X_oos)[:,0]
        pred_oos /= nbags
        i += 1
        print("Fold", i, "mae oos: ", np.mean(abs(pred_oos-y_oos)))
    pred_final /= (nbags*nfolds)
    print(pred_final)
    return pred_final

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

model = Sequential()
model.add(Dense(400, input_dim = data_train.shape[1], init = 'he_normal'))
model.add(PReLU())
model.add(Dropout(0.4))
model.add(Dense(200, init = 'he_normal'))
model.add(PReLU())
model.add(Dropout(0.2))
model.add(Dense(1, init = 'he_normal'))
model.compile(optimizer='adadelta', loss='mae')

#model.compile(optimizer=sgd, loss='mae')
#model.fit(data_train.values, y_train, batch_size=50, nb_epoch=nepochs, shuffle=True)
pred_1 = bag_model_cv(data_train.values, y_train, model)

model = Sequential()
model.add(Dense(600, input_dim = data_train.shape[1], init = 'he_normal'))
model.add(PReLU())
model.add(Dropout(0.4))
model.add(Dense(400, init = 'he_normal'))
model.add(PReLU())
model.add(Dropout(0.2))
model.add(Dense(1, init = 'he_normal'))
model.compile(optimizer='adadelta', loss='mae')

pred_2 = bag_model_cv(data_train.values, y_train, model)

model = Sequential()
model.add(Dense(800, input_dim = data_train.shape[1], init = 'he_normal'))
model.add(PReLU())
model.add(Dropout(0.4))
model.add(Dense(400, init = 'he_normal'))
model.add(PReLU())
model.add(Dropout(0.2))
model.add(Dense(1, init = 'he_normal'))
model.compile(optimizer='adadelta', loss='mae')

pred_3 = bag_model_cv(data_train.values, y_train, model)

model = Sequential()
model.add(Dense(400, input_dim=data_train.shape[1], init='glorot_uniform'))
model.add(PReLU())
model.add(Dropout(0.4))
model.add(Dense(200, init='glorot_uniform'))
model.add(LeakyReLU())
model.add(Dropout(0.4))
model.add(Dense(100, init='glorot_uniform'))
model.add(LeakyReLU())
model.add(Dropout(0.2))
model.add(Dense(1, init = 'glorot_normal'))
model.compile(optimizer='adadelta', loss='mae')

pred_4 = bag_model_cv(data_train.values, y_train, model)

df = pd.DataFrame(test_id, columns = ['id'])
df['nn_pred_1'] = pred_1
df['nn_pred_2'] = pred_2
df['nn_pred_3'] = pred_3
df['nn_pred_4'] = pred_4

df.to_csv('keras_full_stack.csv', index = False)