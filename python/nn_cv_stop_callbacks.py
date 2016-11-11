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

#y_train = np.log(y_train)

test_normal = pd.merge(test['id'].to_frame(), df_normal, on='id', how='inner', sort=False)
train_normal = pd.merge(train['id'].to_frame(), df_normal, on='id', how='inner', sort=False)

train_normal.drop(['id', 'loss'], axis=1, inplace=True)
test_normal.drop(['id', 'loss'], axis=1, inplace=True)

data_test = test_normal
data_train = train_normal

nfolds = 5
kf = KFold(nfolds, shuffle=True)

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0, patience=10, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1)
#checkpoint = ModelCheckpoint('Checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5', save_best_only=True, verbose=1)

df_results = pd.DataFrame(np.arange(0, nfolds, 1), columns=['fold'])
df_results['fold'] += 1
df_results['mae'] = np.zeros(df_results.shape[0])

def model_cv(X, y, nn_model):
    j=0
    pred_final = np.zeros(data_test.shape[0])
    for train_index, test_index in kf.split(X):
        X_model, X_oos = X[train_index], X[test_index]
        y_model, y_oos = y[train_index], y[test_index]
        pred_oos = np.zeros(y_oos.shape[0])
        model.compile(optimizer='adadelta', loss='mae')
        checkpoint = ModelCheckpoint('Checkpoint-Fold:%d.hdf5' % (j+1), save_best_only=True, verbose=1)
        nn_model.fit(X_model, y_model, batch_size=500, nb_epoch=500, 
                     validation_data = (X_oos, y_oos), shuffle=True,
                     callbacks=[early_stopping, reduce_lr, checkpoint])
        
        #pred_oos = nn_model.predict(X_oos)[:,0]
        #load checkpointed best model
        nn_model.load_weights('Checkpoint-Fold:%d.hdf5' % (j+1))
        pred_final += nn_model.predict(data_test.values, batch_size=1000)[:,0]
        #df_results.ix[j,1] = np.mean(abs(pred_oos-y_oos))
        j += 1
    pred_final /= nfolds
    print(pred_final)
    return pred_final

class LossHistory(Callback):
    def on_epoch_end(self, epoch, logs={}):
        print("\n")
        print("Epoch: ", epoch)
        #print(self.params)
        #print(self.model.optimizer.get_config())
    def on_train_end(self, logs={}):
        print(logs)

class LossValidate(Callback):
    def __init__(self, validation_data=(), patience=5):
        super(Callback, self).__init__()
        
        self.patience = patience
        self.X_val, self.y_val = validation_data  #tuple of validation X and y
        self.best = 5000
        self.wait = 0  #counter for patience
        self.best_rounds = 1
        self.counter = 0
    def on_epoch_end(self, epoch, logs={}):
        self.counter +=1
        pred_oos = self.model.predict(self.X_val, verbose=0)[:,0]
        current = np.mean(abs(pred_oos-self.y_val))
        print("\n")
        print("Epoch: %d MAE: %f | Best Mae: %f \n" %(epoch, current, self.best))
        #if improvement over best
        if current < self.best:
            self.best = current
            self.best_rounds=self.counter
            self.wait = 0
        else:
            if self.wait >= self.patience:
                self.model.stop_training = True
                print('Best number of rounds: %d \nMAE: %f \n' % (self.best_rounds, self.best))
            self.wait += 1
            print('Rounds with no Improvement: %d\n' % (self.wait))

        

model = Sequential()
model.add(Dense(400, input_dim = data_train.shape[1], init = 'he_normal'))
model.add(PReLU())
model.add(Dropout(0.4))
model.add(Dense(200, init = 'he_normal'))
model.add(PReLU())
model.add(Dropout(0.2))
model.add(Dense(1, init = 'he_normal'))

#history = LossHistory()
#model.compile(optimizer=sgd, loss='mae')
#loss_validate = LossValidate(validation_data=(data_train.values, y_train), patience=5)
#test = model.fit(data_train.values, y_train, batch_size=500, nb_epoch=2, validation_split = .5, shuffle=True, callbacks=[loss_validate])

#df_results = pd.DataFrame(np.arange(0, nfolds, 1), columns=['fold'])
#df_results['fold'] += 1
#df_results['mae'] = np.zeros(df_results.shape[0])
#df_results['best_round'] = np.zeros(df_results.shape[0])

def model_cv2(X, y, nn_model):
    j=0
    pred_final = np.zeros(data_test.shape[0])
    for train_index, test_index in kf.split(X):
        X_model, X_oos = X[train_index], X[test_index]
        y_model, y_oos = y[train_index], y[test_index]
        loss_validate = LossValidate(validation_data = (X_oos, y_oos), patience=5)
        model.compile(optimizer='adadelta', loss='mae')
        nn_model.fit(X_model, y_model, batch_size=500, nb_epoch=500, 
                     shuffle=True,
                     callbacks=[loss_validate])
        print(loss_validate.best_rounds)
        pred_final += nn_model.predict(data_test.values, batch_size=1000)[:,0]
        #pred_oos = nn_model.predict(X_oos)[:,0]
        #df_results.ix[j,1] = np.mean(abs(pred_oos-y_oos))
        #df_results.ix[j,2] = loss_validate.best_rounds
        #j += 1
    pred_final /= nfolds
    print(pred_final)
    return pred_final

#test = model.fit(data_train.values, y_train, batch_size=500, nb_epoch=1000, 
#                 validation_split = .5, shuffle=True, callbacks=[early_stopping, reduce_lr])
#test.history

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model = Sequential()
model.add(Dense(400, input_dim = data_train.shape[1], init = 'he_normal'))
model.add(PReLU())
model.add(Dropout(0.4))
model.add(Dense(200, init = 'he_normal'))
model.add(PReLU())
model.add(Dropout(0.2))
model.add(Dense(1, init = 'he_normal'))

pred_1 = model_cv(data_train.values, y_train, model)

model = Sequential()
model.add(Dense(600, input_dim = data_train.shape[1], init = 'he_normal'))
model.add(PReLU())
model.add(Dropout(0.4))
model.add(Dense(400, init = 'he_normal'))
model.add(PReLU())
model.add(Dropout(0.2))
model.add(Dense(1, init = 'he_normal'))

pred_2 = model_cv(data_train.values, y_train, model)

model = Sequential()
model.add(Dense(800, input_dim = data_train.shape[1], init = 'he_normal'))
model.add(PReLU())
model.add(Dropout(0.4))
model.add(Dense(400, init = 'he_normal'))
model.add(PReLU())
model.add(Dropout(0.2))
model.add(Dense(1, init = 'he_normal'))

pred_3 = model_cv(data_train.values, y_train, model)

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

pred_4 = model_cv(data_train.values, y_train, model)

df = pd.DataFrame(test_id, columns = ['id'])
df['nn_pred_1'] = pred_1
df['nn_pred_2'] = pred_2
df['nn_pred_3'] = pred_3
df['nn_pred_4'] = pred_4

df.to_csv('keras_cv_full.csv', index = False)
