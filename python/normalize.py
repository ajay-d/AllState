import numpy as np
import pandas as pd

data_train = pd.read_csv('train_python_a.csv.gz', compression="gzip")
data_test = pd.read_csv('train_python_b.csv.gz', compression="gzip")

#frames = [data_train, data_test]
#data_train = pd.concat(frames, ignore_index=True)

train_id = data_train['id'].values
test_id = data_test['id'].values

train_normal = pd.DataFrame(train_id, columns = ['id'])
for f in data_train.columns:
    #if 'cat' in f:
        #print(f)
    if 'cat' in f:
        d = pd.get_dummies(data_train[f])
        frames = [train_normal, d]
        train_normal = pd.concat(frames, axis=1)
        
test_normal = pd.DataFrame(test_id, columns = ['id'])
for f in data_test.columns:
    if 'cat' in f:
        d = pd.get_dummies(data_test[f])
        frames = [test_normal, d]
        test_normal = pd.concat(frames, axis=1)
        


pd.get_dummies(data_train['cat1']).head()
data_train['cat1'].head()

pd.get_dummies(data_train['cat116']).head()
data_train['cat116'].head()

train_normal.shape
test_normal.shape

f_num = [f for f in data_train.columns if 'cont' in f]
data_train[f_num].head()
s_train = (data_train[f_num] - data_train[f_num].mean()) / data_train[f_num].std()
frames = [train_normal, s_train]
train_normal = pd.concat(frames, axis=1)

s_test = (data_test[f_num] - data_test[f_num].mean()) / data_test[f_num].std()
frames = [test_normal, s_test]
test_normal = pd.concat(frames, axis=1)

train_normal.shape
test_normal.shape

#################
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

df_normal.index
df_normal.shape
df_normal.head
df_normal.columns
[f for f in all_data.columns if 'cat' in f]

data_test

df_normal[df_normal['id']==479622]

#test_new = df_normal.loc[:,'id'] = data_test['id']
#test_new = df_normal.join(data_test['id'].to_frame(), on='id', how='inner', rsuffix="_")
test_new = pd.merge(df_normal, data_test['id'].to_frame(), on='id', how='inner')
data_test.shape
test_new.shape

test_new.head()

