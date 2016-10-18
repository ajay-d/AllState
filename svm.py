# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 15:43:57 2016

@author: adeonari
"""

import os
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor

os.chdir('c:/users/adeonari/downloads/allstate')
data = pd.read_csv('train_recode.csv.gz', compression="gzip")


data.head()
data.info()
data.describe()
data.columns

y_train = data['loss'].ravel()
y_train = np.log(y_train)
data.drop(['id', 'loss'], axis=1, inplace=True)

etr = ExtraTreesRegressor(criterion='mae')
etr.fit(data, y_train)

y_etr = ert.predict(data)

svr_rbf = svm.SVR(kernel='rbf')
svr_lin = svm.SVR(kernel='linear')
nu_rbf = svm.NuSVR(kernel='rbf')
nu_lin = svm.NuSVR(kernel='linear')
lin = svm.LinearSVR()


