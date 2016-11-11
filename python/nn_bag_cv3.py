import numpy as np
np.random.seed(666)

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
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

data_train = pd.read_csv('train_python_a.csv.gz', compression="gzip")
data_test = pd.read_csv('train_python_b.csv.gz', compression="gzip")
frames = [data_train, data_test]
data_train = pd.concat(frames, ignore_index=True)
data_test = pd.read_csv('test_python.csv.gz', compression="gzip")

##Mean normalize data
train_id = data_train['id'].values
test_id = data_test['id'].values

y_train = data_train['loss'].ravel()
y_test = data_test['loss'].ravel()

#y_train = np.log(y_train)

test_normal = pd.merge(data_test['id'].to_frame(), df_normal, on='id', how='inner', sort=False)
train_normal = pd.merge(data_train['id'].to_frame(), df_normal, on='id', how='inner', sort=False)

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
    pred_final = np.zeros(y_test.shape[0])
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

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    
model = Sequential()
model.add(Dense(400, input_dim = data_train.shape[1], init = 'he_normal'))
model.add(PReLU())
model.add(Dropout(0.4))
model.add(Dense(200, init = 'he_normal'))
model.add(PReLU())
model.add(Dropout(0.2))
model.add(Dense(1, init = 'he_normal'))
model.compile(optimizer='adadelta', loss='mae')

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

model = Sequential()
model.add(Dense(400, input_dim = data_train.shape[1], init = 'he_normal'))
model.add(PReLU())
model.add(Dropout(0.4))
model.add(Dense(200, init = 'he_normal'))
model.add(PReLU())
model.add(Dropout(0.2))
model.add(Dense(50, init = 'he_normal'))
model.add(PReLU())
model.add(Dropout(0.2))
model.add(Dense(1, init = 'he_normal'))
model.compile(optimizer='adadelta', loss='mae')

pred_5 = bag_model_cv(data_train.values, y_train, model)

##
print(np.mean(abs(pred_1-y_test)))
##
print(np.mean(abs(pred_2-y_test)))
##
print(np.mean(abs(pred_3-y_test)))
##
print(np.mean(abs(pred_4-y_test)))
##
print(np.mean(abs(pred_5-y_test)))


