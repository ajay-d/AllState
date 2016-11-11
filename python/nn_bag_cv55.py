import numpy as np
np.random.seed(666)

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
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

def create_model(m):
    if m==1:
        model = Sequential()
        model.add(Dense(400, input_dim = data_train.shape[1], init = 'he_normal'))
        model.add(PReLU())
        model.add(Dropout(0.4))
        model.add(Dense(200, init = 'he_normal'))
        model.add(PReLU())
        model.add(Dropout(0.2))
        model.add(Dense(1, init = 'he_normal'))
        model.compile(optimizer='adadelta', loss='mae')
        return model
    if m==2:
        model = Sequential()
        model.add(Dense(400, input_dim = data_train.shape[1], init = 'he_normal'))
        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Dense(200, init = 'he_normal'))
        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(1, init = 'he_normal'))
        model.compile(optimizer='adadelta', loss='mae')
        return model
    if m==3:
        model = Sequential()
        model.add(Dense(600, input_dim = data_train.shape[1], init = 'he_normal'))
        model.add(PReLU())
        model.add(Dropout(0.4))
        model.add(Dense(400, init = 'he_normal'))
        model.add(PReLU())
        model.add(Dropout(0.2))
        model.add(Dense(1, init = 'he_normal'))
        model.compile(optimizer='adadelta', loss='mae')
        return model
    if m==4:
        model = Sequential()
        model.add(Dense(800, input_dim = data_train.shape[1], init = 'he_normal'))
        model.add(PReLU())
        model.add(Dropout(0.4))
        model.add(Dense(400, init = 'he_normal'))
        model.add(PReLU())
        model.add(Dropout(0.2))
        model.add(Dense(1, init = 'he_normal'))
        model.compile(optimizer='adadelta', loss='mae')
        return model
    if m==5:
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
        return model        

nbags = 5
nfolds = 5
nepochs = 55

##To store all CV results
df_results = pd.DataFrame(np.arange(0, nfolds, 1), columns=['fold'])
df_results['fold'] += 1
df_results['key'] = 1
df_results['mae'] = np.zeros(df_results.shape[0])

df_models = pd.DataFrame(np.arange(1, 6, 1), columns=['model'])
df_models['key'] = 1

df_results = pd.merge(df_models, df_results, on='key', how='inner', sort=True)
df_results.drop(['key'], axis=1, inplace=True)

kf = KFold(nfolds, shuffle=True)

def bag_model_cv(X, y, model_n):
    i = 0
    #pred_final = np.zeros(y_test.shape[0])
    pred_final = np.zeros(data_test.shape[0])
    print(pred_final)
    for train_index, test_index in kf.split(X):
        X_model, X_oos = X[train_index], X[test_index]
        y_model, y_oos = y[train_index], y[test_index]
        #sample_index = np.arange(X_model.shape[0])
        pred_oos = np.zeros(y_oos.shape[0])
        for j in range(nbags):
            nn_model = create_model(model_n)
            nn_model.fit(X_model, y_model, batch_size=500, nb_epoch=nepochs, shuffle=True)
            pred_final += nn_model.predict(data_test.values, batch_size=1000)[:,0]
            pred_oos += nn_model.predict(X_oos)[:,0]
        pred_oos /= nbags
        i += 1
        df_results.loc[(df_results['fold']==i) & (df_results['model']==model_n), 'mae'] = np.mean(abs(pred_oos - y_oos))
        print("Fold", i, "mae oos: ", np.mean(abs(pred_oos-y_oos)))
    pred_final /= (nbags*nfolds)
    print(pred_final)
    return pred_final


pred_1 = bag_model_cv(data_train.values, y_train, 1)
pred_2 = bag_model_cv(data_train.values, y_train, 2)
pred_3 = bag_model_cv(data_train.values, y_train, 3)
pred_4 = bag_model_cv(data_train.values, y_train, 4)
pred_5 = bag_model_cv(data_train.values, y_train, 5)

df = pd.DataFrame(test_id, columns = ['id'])
df['nn_pred_1'] = pred_1
df['nn_pred_2'] = pred_2
df['nn_pred_3'] = pred_3
df['nn_pred_4'] = pred_4
df['nn_pred_5'] = pred_5

df.to_csv('keras_full_55.csv', index = False)
df_results

