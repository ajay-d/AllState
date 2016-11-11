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
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback

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
shift = 200
y_train = np.log(y_train + shift)

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
        return model
    if m==2:
        model = Sequential()
        model.add(Dense(600, input_dim = data_train.shape[1], init = 'he_normal'))
        model.add(PReLU())
        model.add(Dropout(0.4))
        model.add(Dense(400, init = 'he_normal'))
        model.add(PReLU())
        model.add(Dropout(0.2))
        model.add(Dense(1, init = 'he_normal'))
        return model
    if m==3:
        model = Sequential()
        model.add(Dense(800, input_dim = data_train.shape[1], init = 'he_normal'))
        model.add(PReLU())
        model.add(Dropout(0.4))
        model.add(Dense(400, init = 'he_normal'))
        model.add(PReLU())
        model.add(Dropout(0.2))
        model.add(Dense(1, init = 'he_normal'))
        return model
    if m==4:
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
        return model


nfolds = 5
kf = KFold(nfolds, shuffle=True)

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0, patience=10, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

def model_cv(X, y, model_n):
    j = 1
    pred_final = np.zeros(data_test.shape[0])
    for train_index, test_index in kf.split(X):
        X_model, X_oos = X[train_index], X[test_index]
        y_model, y_oos = y[train_index], y[test_index]
        pred_oos = np.zeros(y_oos.shape[0])
        nn_model = create_model(model_n)
        nn_model.compile(optimizer='RMSprop', loss='mse')
        checkpoint = ModelCheckpoint('Checkpoint_rms-Fold-%d.hdf5' % (j), save_best_only=True, verbose=1)
        nn_model.fit(X_model, y_model, batch_size=500, nb_epoch=500, 
                     validation_data = (X_oos, y_oos), shuffle=True,
                     callbacks=[early_stopping, reduce_lr, checkpoint])
        nn_model.load_weights('Checkpoint_rms-Fold-%d.hdf5' % (j))
        pred_final += nn_model.predict(data_test.values, batch_size=1000)[:,0]
        pred_oos = nn_model.predict(X_oos)[:,0]
        print("Fold", j, "mae oos: ", np.mean(abs(np.exp(pred_oos)-200-np.exp(y_oos)-200)))
        j += 1
    pred_final = np.exp(pred_final/nfolds) - shift
    return pred_final


#m1 = create_model(1)
#m1.summary()
#m1.get_config()
#m1.get_weights()
#m1.layers
#m1.layers[1].get_weights()

pred_1 = model_cv(data_train.values, y_train, 1)
pred_2 = model_cv(data_train.values, y_train, 2)
pred_3 = model_cv(data_train.values, y_train, 3)
pred_4 = model_cv(data_train.values, y_train, 4)

df = pd.DataFrame(test_id, columns = ['id'])
df['nn_pred_1'] = pred_1
df['nn_pred_2'] = pred_2
df['nn_pred_3'] = pred_3
df['nn_pred_4'] = pred_4

df.to_csv('keras_cv_full_rms.csv', index = False)