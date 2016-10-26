import os
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
from keras.optimizers import SGD

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

y_train = np.log(y_train)

test_normal = pd.merge(df_normal, data_test['id'].to_frame(), on='id', how='inner')
train_normal = pd.merge(df_normal, data_train['id'].to_frame(), on='id', how='inner')

train_normal.drop(['id', 'loss'], axis=1, inplace=True)
test_normal.drop(['id', 'loss'], axis=1, inplace=True)

data_test = test_normal
data_train = train_normal

np.random.seed(666)

bags = 2
nepochs = 10
def bag_model(X, Y, nn_model):
    sample_index = np.arange(X.shape[0])
    pred = np.zeros(y_test.shape[0])
    for j in range(bags):
        np.random.shuffle(sample_index)
        x_bag = X[sample_index]
        y_bag = Y[sample_index]
        nn_model.fit(x_bag, y_bag, nb_epoch=nepochs)
        pred += model.predict(data_test.values)[:,0]
    pred /= bags
    return pred

model = Sequential()
model.add(Dense(400, input_dim = data_train.shape[1], init = 'he_normal'))
model.add(PReLU())
model.add(Dropout(0.4))
model.add(Dense(200, init = 'he_normal'))
model.add(PReLU())
model.add(Dropout(0.2))
model.add(Dense(1, init = 'he_normal'))
model.compile(loss = 'mae', optimizer = 'adadelta')
        
pred_1 = bag_model(data_train.values, y_train, model)
print(np.mean(abs(np.exp(pred_1[:,0]-y_test))))

model.compile(optimizer='rmsprop', loss='mae')
model.compile(optimizer='adadelta', loss='mae')
model.compile(optimizer='adagrad', loss='mae')
model.compile(optimizer='adam', loss='mae')
model.compile(optimizer='adamax', loss='mae')
model.compile(optimizer='nadam', loss='mae')

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='mae')

df = pd.DataFrame(test_id, columns = ['id'])
df['nn_pred_1'] = pred_1
df['nn_pred_2'] = pred_2
df['nn_pred_3'] = pred_3

df.to_csv('nn_preds_bag1.csv', index=False)
