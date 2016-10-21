import os
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import SGD

#https://github.com/fchollet/keras/issues/3857
tf.python.control_flow_ops = tf

data_train = pd.read_csv('train_python.csv.gz', compression="gzip")
data_test = pd.read_csv('test_python.csv.gz', compression="gzip")

np.random.seed(666)

train_id = data_train['id'].values
test_id = data_test['id'].values

y_train = data_train['loss'].ravel()
y_test = data_test['loss'].ravel()

data_train.drop(['id', 'loss'], axis=1, inplace=True)
data_test.drop(['id', 'loss'], axis=1, inplace=True)

model = Sequential()
model.add(Dense(400, input_dim = data_train.shape[1], init = 'he_normal'))
model.add(PReLU())
model.add(Dropout(0.4))
model.add(Dense(200, init = 'he_normal'))
model.add(PReLU())
model.add(Dropout(0.2))
model.add(Dense(1, init = 'he_normal'))
model.compile(optimizer = 'adadelta', loss = 'mae')
model.fit(data_train.values, y_train, nb_epoch=5)
pred_1 = model.predict(data_train.values)

model = Sequential()
model.add(Dense(10, input_dim=data_train.shape[1], init='uniform', activation='relu'))
model.add(Dense(1, init='normal'))
model.compile(optimizer='adagrad', loss='mae')
model.fit(data_train.values, y_train, nb_epoch=5)

pred_2 = model.predict(data_train.values)

model = Sequential()
model.add(Dense(10, input_dim=data_train.shape[1], init='uniform', activation='relu'))
model.add(Dense(1, init='normal'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='mae')
model.fit(data_train.values, y_train, nb_epoch=5)

pred_3 = model.predict(data_train.values)

model = Sequential()
model.add(Dense(64, input_dim=data_train.shape[1], init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(64, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.2))
model.add(Dense(1, init = 'he_normal'))
model.compile(optimizer='adadelta', loss='mae')
model.fit(data_train.values, y_train, nb_epoch=5)

pred_4 = model.predict(data_train.values)

model = Sequential()
model.add(Dense(64, input_dim=data_train.shape[1], init='uniform'))
model.add(BatchNormalization())
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(64, init='uniform'))
model.add(BatchNormalization())
model.add(Activation('tanh'))
model.add(Dropout(0.2))
model.add(Dense(1, init = 'he_normal'))
model.compile(optimizer='adadelta', loss='mae')
model.fit(data_train.values, y_train, nb_epoch=5)

pred_5 = model.predict(data_train.values)

model = Sequential()
model.add(Dense(100, input_dim=data_train.shape[1], init='uniform', activation='linear'))
model.add(PReLU())
model.add(Dropout(0.5))
model.add(Dense(100, init='uniform', activation='linear'))
model.add(LeakyReLU(alpha=.001))
model.add(Dropout(0.2))
model.add(Dense(100, init='uniform', activation='linear'))
model.add(LeakyReLU(alpha=.001))
model.add(Dropout(0.2))
model.add(Dense(1, init = 'he_normal'))
model.compile(optimizer='adadelta', loss='mae')
model.fit(data_train.values, y_train, nb_epoch=5)

pred_6 = model.predict(data_train.values)

model = Sequential()
model.add(Dense(100, input_dim=data_train.shape[1], init='uniform', activation='linear'))
model.add(PReLU())
model.add(Dense(100, init='uniform', activation='linear'))
model.add(LeakyReLU(alpha=.001))
model.add(Dense(100, init='uniform', activation='linear'))
model.add(LeakyReLU(alpha=.001))
model.add(Dense(1, init = 'he_normal'))
model.compile(optimizer='adadelta', loss='mae')
model.fit(data_train.values, y_train, nb_epoch=5)

pred_7 = model.predict(data_train.values)

##Maybe add more epochs
df = pd.DataFrame(train_id, columns = ['id'])
df['nn_pred_1'] = pred_1
df['nn_pred_2'] = pred_2
df['nn_pred_3'] = pred_3
df['nn_pred_4'] = pred_4
df['nn_pred_5'] = pred_5
df['nn_pred_6'] = pred_6
df['nn_pred_7'] = pred_7

df.to_csv('nn_preds_train_5epochs.csv', index=False)

