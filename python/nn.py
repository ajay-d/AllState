import os
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
from keras.optimizers import SGD

tf.python.control_flow_ops = tf

#os.chdir('c:/users/adeonari/downloads/allstate')
#os.chdir('/users/ajay/downloads/allstate')
data = pd.read_csv('train_recode.csv.gz', compression="gzip")

data_train = pd.read_csv('train_python.csv.gz', compression="gzip")
data_test = pd.read_csv('test_python.csv.gz', compression="gzip")

np.random.seed(666)

test_id = data_test['id'].values

y_train = data_train['loss'].ravel()
data_train.drop(['id', 'loss'], axis=1, inplace=True)
data_test.drop(['id', 'loss'], axis=1, inplace=True)

bags = 5
nepochs = 55
def bag_model(X, Y, nn_model):
    sample_index = np.arange(X.shape[0])
    pred = np.zeros(Y.shape[0])
    for j in range(bags):
        np.random.shuffle(sample_index)
        x_bag = X[sample_index]
        y_bag = Y[sample_index]
        nn_model.fit(x_bag, y_bag, nb_epoch=nepochs)
        pred += model.predict(data_test.values)
        
bag_model(data_train.values, y_train, model, 3, 3)

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

pred_1 = model.predict(data_test.values)

df = pd.DataFrame(test_id, columns = ['id'])
df['nn_pred_1'] = pred_1
df['nn_pred_2'] = pred_2
df['nn_pred_3'] = pred_3

df.to_csv('nn_preds.csv', index=False)
