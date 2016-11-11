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
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback

#https://github.com/fchollet/keras/issues/3857
tf.python.control_flow_ops = tf

train = pd.read_csv('train_recode_factor.csv.gz', compression="gzip")
test = pd.read_csv('test_recode_factor.csv.gz', compression="gzip")
train.shape
test.shape

train_id = train['id'].values
test_id = test['id'].values

y_train = train['loss'].ravel()

shift = 200
y_train_log = np.log(y_train + shift)

X = train.drop(['id', 'loss'], axis=1)
X_test = test.drop(['id'], axis=1)

nfolds = 5
kf = KFold(nfolds, shuffle=True)

etr_1 = ExtraTreesRegressor(n_estimators=5, max_depth=None, n_jobs=-1)
etr_2 = ExtraTreesRegressor(n_estimators=10, max_depth=None, n_jobs=-1)
etr_3 = ExtraTreesRegressor(n_estimators=5, max_depth=4, n_jobs=-1)
etr_4 = ExtraTreesRegressor(n_estimators=10, max_depth=5, n_jobs=-1)

boost_etr_1 = AdaBoostRegressor(etr_1, n_estimators = 25, random_state=333)
boost_etr_2 = AdaBoostRegressor(etr_2, n_estimators = 25, random_state=333)
boost_etr_3 = AdaBoostRegressor(etr_1, n_estimators = 50, random_state=444, learning_rate=.5)
boost_etr_4 = AdaBoostRegressor(etr_2, n_estimators = 50, random_state=444, learning_rate=.5)
boost_etr_5 = AdaBoostRegressor(etr_3, n_estimators = 100, random_state=555, learning_rate=.5)
boost_etr_6 = AdaBoostRegressor(etr_4, n_estimators = 100, random_state=555, learning_rate=.5)

pred_final = np.zeros(test.shape[0])
models = [boost_etr_1, boost_etr_2, boost_etr_3, boost_etr_4, boost_etr_5, boost_etr_6]

#To store stacked values
train_stack = pd.DataFrame(train_id, columns = ['id'])
test_stack = pd.DataFrame(test_id, columns = ['id'])

for i in np.arange(len(models))+1:
    train_stack["xtra_%d" % (i)] = np.zeros(train_stack.shape[0])
    test_stack["xtra_%d" % (i)] = np.zeros(test_stack.shape[0])

for train_index, test_index in kf.split(X):
    X_model, X_oos = X.values[train_index], X.values[test_index]
    y_model, y_oos = y_train_log[train_index], y_train_log[test_index]
    col = 1
    fold = 1
    for i in models:
        i.fit(X_model, y_model)
        train_stack.ix[test_index, col] = i.predict(X_oos)
        test_stack.ix[:, col] += i.predict(X_test.values)
        pred_oos = i.predict(X_oos)
        print("Fold", fold, "model", col, "mae oos: ", np.mean(abs(np.exp(pred_oos)-200-np.exp(y_oos)-200)))
        col += 1
    fold += 1

for i in np.arange(len(models))+1:
    test_stack.ix[:, i] /= nfolds

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
test_normal = pd.merge(test['id'].to_frame(), df_normal, on='id', how='inner', sort=False)
train_normal = pd.merge(train['id'].to_frame(), df_normal, on='id', how='inner', sort=False)

##Normalize stacked values
all_stack = pd.concat([train_stack, test_stack], ignore_index=True)
all_stack_norm = pd.DataFrame(all_stack, columns = ['id'])
f_num = [f for f in all_stack.columns if 'xtra' in f]
s = (all_stack[f_num] - all_stack[f_num].mean()) / all_stack[f_num].std()
frames = [all_stack_norm, s]
all_stack_norm = pd.concat(frames, axis=1)

##Add stacking values
test_normal = pd.merge(test_normal, all_stack_norm, on='id', how='inner', sort=False)
train_normal = pd.merge(train_normal, all_stack_norm, on='id', how='inner', sort=False)

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
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1)

def model_cv(X, y, model_n):
    j = 1
    pred_final = np.zeros(data_test.shape[0])
    for train_index, test_index in kf.split(X):
        X_model, X_oos = X[train_index], X[test_index]
        y_model, y_oos = y[train_index], y[test_index]
        pred_oos = np.zeros(y_oos.shape[0])
        nn_model = create_model(model_n)
        nn_model.compile(optimizer='adadelta', loss='mae')
        #nn_model.compile(optimizer='RMSprop', loss='mae')
        checkpoint = ModelCheckpoint('Checkpoint_ada-Fold-%d.hdf5' % (j), save_best_only=True, verbose=1)
        #checkpoint = ModelCheckpoint('Checkpoint_rms-Fold-%d.hdf5' % (j), save_best_only=True, verbose=1)
        nn_model.fit(X_model, y_model, batch_size=500, nb_epoch=500, 
                     validation_data = (X_oos, y_oos), shuffle=True,
                     callbacks=[early_stopping, checkpoint])
        nn_model.load_weights('Checkpoint_ada-Fold-%d.hdf5' % (j))
        #nn_model.load_weights('Checkpoint_rms-Fold-%d.hdf5' % (j))
        pred_final += nn_model.predict(data_test.values, batch_size=1000)[:,0]
        pred_oos = nn_model.predict(X_oos)[:,0]
        print("Fold", j, "mae oos: ", np.mean(abs(pred_oos-y_oos)))
        j += 1
    pred_final /= nfolds
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

df.to_csv('keras_cv_full_stack.csv', index = False)