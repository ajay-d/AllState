import os
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor

#os.chdir('c:/users/adeonari/downloads/allstate')
#os.chdir('/users/ajay/downloads/allstate')
data = pd.read_csv('train_recode.csv.gz', compression="gzip")

data_train = pd.read_csv('train_python.csv.gz', compression="gzip")
data_test = pd.read_csv('train_python.csv.gz', compression="gzip")

data.head()
data.info()
data.describe()
data.columns

y_train = data_train['loss'].ravel()
#y_train = np.log(y_train)
data_train.drop(['id', 'loss'], axis=1, inplace=True)
data_test.drop(['id', 'loss'], axis=1, inplace=True)

etr_1 = ExtraTreesRegressor(criterion='mae')
etr_2 = ExtraTreesRegressor()

#etr_1.fit(data_train, y_train)
etr_2.fit(data_train, y_train)
y_etr = etr_2.predict(data_test)

svr_rbf = svm.SVR(kernel='rbf')
svr_lin = svm.SVR(kernel='linear')
nu_rbf = svm.NuSVR(kernel='rbf')
nu_lin = svm.NuSVR(kernel='linear')
lin = svm.LinearSVR()

svr_rbf.fit(data_train, y_train)
y_svr_rbf = svr_rbf.predict(data_test)

df = pd.DataFrame({'id': data_test['loss'].ravel(),
    'loss': y_svr_rbf})
df.to_csv('sk_preds_validate.csv', index=False)
})