import os
import pandas as pd
import numpy as np
from sklearn import svm

#os.chdir('c:/users/adeonari/downloads/allstate')
#os.chdir('/users/ajay/downloads/allstate')
data = pd.read_csv('train_recode.csv.gz', compression="gzip")

data_train = pd.read_csv('train_python.csv.gz', compression="gzip")
data_test = pd.read_csv('train_python.csv.gz', compression="gzip")

np.random.seed(666)

test_id = data_test['id'].values

y_train = data_train['loss'].ravel()
#y_train = np.log(y_train)
data_train.drop(['id', 'loss'], axis=1, inplace=True)
data_test.drop(['id', 'loss'], axis=1, inplace=True)

svr_rbf = svm.SVR(kernel='rbf')
svr_lin = svm.SVR(kernel='linear')
nu_rbf = svm.NuSVR(kernel='rbf')
nu_lin = svm.NuSVR(kernel='linear')
lin = svm.LinearSVR()

svr_rbf.fit(data_train, y_train)
y_svr_rbf = svr_rbf.predict(data_test)

#df = pd.DataFrame(pred_1[:,0], columns = ['pred_1'])
#df['pred_2'] = pred_2[:,0]

df = pd.DataFrame(test_id, columns = ['id'])
df['svm_1'] = y_svr_rbf

df.to_csv('svm_preds.csv', index=False)