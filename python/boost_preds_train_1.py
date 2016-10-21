import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

#os.chdir('c:/users/adeonari/downloads/allstate')
#os.chdir('/users/ajay/downloads/allstate')
data = pd.read_csv('train_recode.csv.gz', compression="gzip")

data_train = pd.read_csv('train_python.csv.gz', compression="gzip")
data_test = pd.read_csv('train_python.csv.gz', compression="gzip")

data.head()
data.info()
data.describe()
data.columns

np.random.seed(666)

train_id = data_train['id'].values
test_id = data_test['id'].values

y_train = data_train['loss'].ravel()
y_test = data_test['loss'].ravel()

data_train.drop(['id', 'loss'], axis=1, inplace=True)
data_test.drop(['id', 'loss'], axis=1, inplace=True)

#etr_1 = ExtraTreesRegressor(criterion='mae')
etr_1 = ExtraTreesRegressor(n_estimators=10, n_jobs=-1)
etr_2 = ExtraTreesRegressor(n_estimators=50, n_jobs=-1)
etr_3 = ExtraTreesRegressor(n_estimators=100, n_jobs=-1)

reg_1 = DecisionTreeRegressor(max_depth=4)
boost_1 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators = 300, random_state=666)

boost_etr_1 = AdaBoostRegressor(etr_1, n_estimators = 100, random_state=777, learning_rate=.01)
boost_etr_2 = AdaBoostRegressor(etr_2, n_estimators = 50, random_state=777, learning_rate=.01)
boost_etr_3 = AdaBoostRegressor(etr_3, n_estimators = 25, random_state=777, learning_rate=.01)
boost_etr_4 = AdaBoostRegressor(ExtraTreesRegressor(max_depth=4), n_estimators = 300, random_state=777, learning_rate=.01)
boost_etr_5 = AdaBoostRegressor(etr_1, n_estimators = 100, random_state=333)
boost_etr_6 = AdaBoostRegressor(etr_3, n_estimators = 5, random_state=222)

#100, 200, 500, 1000
rf_1 = RandomForestRegressor(n_estimators=100, n_jobs=-1)
rf_2 = RandomForestRegressor(n_estimators=200, n_jobs=-1)
rf_3 = RandomForestRegressor(n_estimators=500, n_jobs=-1)
rf_4 = RandomForestRegressor(n_estimators=1000, n_jobs=-1)

#5, 10, 25, 50, 100
boost_rf_1 = AdaBoostRegressor(rf_1, n_estimators = 100, random_state=111)
boost_rf_2 = AdaBoostRegressor(rf_2, n_estimators = 50, random_state=111)
boost_rf_3 = AdaBoostRegressor(rf_3, n_estimators = 25, random_state=111)
boost_rf_4 = AdaBoostRegressor(rf_4, n_estimators = 10, random_state=111)

etr_1.fit(data_train, y_train)
etr_2.fit(data_train, y_train)
etr_3.fit(data_train, y_train)

reg_1.fit(data_train, y_train)
boost_1.fit(data_train, y_train)

boost_etr_1.fit(data_train, y_train)
boost_etr_2.fit(data_train, y_train)
boost_etr_3.fit(data_train, y_train)
boost_etr_4.fit(data_train, y_train)
boost_etr_5.fit(data_train, y_train)
boost_etr_6.fit(data_train, y_train)

rf_1.fit(data_train, y_train)
rf_2.fit(data_train, y_train)
rf_3.fit(data_train, y_train)
rf_4.fit(data_train, y_train)

boost_rf_1.fit(data_train, y_train)
boost_rf_2.fit(data_train, y_train)
boost_rf_3.fit(data_train, y_train)
boost_rf_4.fit(data_train, y_train)

df = pd.DataFrame(train_id, columns = ['id'])
df['xtra_pred_1'] = etr_1.predict(data_train)
df['xtra_pred_2'] = etr_2.predict(data_train)
df['xtra_pred_3'] = etr_3.predict(data_train)
df['reg_pred_1'] = reg_1.predict(data_train)
df['boost_pred_1'] = boost_1.predict(data_train)
df['boost_etr_pred_1'] = boost_etr_1.predict(data_train)
df['boost_etr_pred_2'] = boost_etr_2.predict(data_train)
df['boost_etr_pred_3'] = boost_etr_3.predict(data_train)
df['boost_etr_pred_4'] = boost_etr_4.predict(data_train)
df['boost_etr_pred_5'] = boost_etr_5.predict(data_train)
df['boost_etr_pred_6'] = boost_etr_6.predict(data_train)
df['rf_pred_1'] = rf_1.predict(data_train)
df['rf_pred_2'] = rf_2.predict(data_train)
df['rf_pred_3'] = rf_3.predict(data_train)
df['rf_pred_4'] = rf_4.predict(data_train)
df['boost_rf_pred_1'] = boost_rf_1.predict(data_train)
df['boost_rf_pred_2'] = boost_rf_2.predict(data_train)
df['boost_rf_pred_3'] = boost_rf_3.predict(data_train)
df['boost_rf_pred_4'] = boost_rf_4.predict(data_train)

df.to_csv('trees_preds_train.csv', index=False)