import os
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
from keras.optimizers import SGD

#os.chdir('c:/users/adeonari/downloads/allstate')
#os.chdir('/users/ajay/downloads/allstate')
data = pd.read_csv('train_recode.csv.gz', compression="gzip")

data_train = pd.read_csv('train_python.csv.gz', compression="gzip")
data_test = pd.read_csv('train_python.csv.gz', compression="gzip")

np.random.seed(666)

y_train = data_train['loss'].ravel()
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
model.compile(loss = 'mae', optimizer = 'adadelta')

model.fit(data_train.values, y_train, nb_epoch=50)

model.add(Activation("relu"))
model.add(Dense(output_dim=10))
model.add(Activation('tanh'))

model.compile(optimizer='rmsprop', loss = 'mae')
model.compile(optimizer='adadelta', loss = 'mae')
model.compile(optimizer='adagrad', loss='mae')

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='mae')

model.fit(data_train.values, y_train, nb_epoch=5)

model = Sequential()
model.add(Dense(10, input_dim=data_train.shape[1], init='uniform', activation='relu'))
model.add(Dense(1, init='normal'))
model.compile(optimizer='adagrad', loss='mae')
model.fit(data_train.values, y_train, nb_epoch=5)

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

pred_1 = model.predict(data_test.values)

