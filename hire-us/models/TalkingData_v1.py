# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import gc

from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# print(os.listdir("../"))
path = "../data"
try:  
    os.mkdir(path)
except OSError:  
    print ("Creation of the directory %s failed" % path)
else:  
    print ("Successfully created the directory %s " % path)
    
# Any results you write to the current directory are saved as output.
print('Loading the data')

dtypes = {  'ip': 'uint16',
            'app': 'uint16',
            'device': 'uint16',
            'os': 'uint16',
            'channel': 'uint16',
            'is_attributed': 'uint8'}
features = ['ip', 'app', 'device', 'os', 'channel', 'click_hour']

# Using RandomUnderSampler to deal with imbalanced data. Since data is pretty big undersampling semmed to be a better choice compare to oversampling
sampling_strategy = 0.5
sampler = RandomUnderSampler(sampling_strategy=sampling_strategy)

X_train = None
y_train = None
i = 0

# Processing in chunks since othervise it requires a lot of memory
for train in pd.read_csv('../input/train.csv', dtype=dtypes, parse_dates=['click_time', 'attributed_time'], chunksize=10000000):
    print("Processing chunk #{}".format(i))
    i+=1
    train['click_hour'] = train['click_time'].dt.hour

    X, y = sampler.fit_resample(train[features], train['is_attributed'])
    X = pd.DataFrame(X, columns=features)
    y = pd.Series(y)

    X_train = pd.concat([X_train,X])
    y_train = pd.concat([y_train,y])

print('Finished data sampling')

# Using GradientBoostingClassifier with parameters obtained from GridSearch. Tried RandomForestClassifier and XGBClassifier 
# but GradientBoostingClassifier gave the best results so far. Probably need to play with parameters more.
# For GradientBoostingClassifier there are a few more options to play with, but they require more time.
# parameters = {
#     "loss":["deviance"],
#     "learning_rate": [0.1, 0.15, 0.2],
#     "max_depth":[3,5,8],
#     "subsample":[0.5, 1.0]
#     }
# clf = GridSearchCV(GradientBoostingClassifier(random_state=0), parameters, cv=3, n_jobs=-1, scoring='roc_auc').fit(X_train_ohe, y_train_usl)
# also tried OneHotEncoder for [ 'app', 'device', 'os', 'channel'] since the data is categorical but for some reason results were not better, 
# still need to play with it.

# from sklearn.preprocessing import OneHotEncoder

# ohe = OneHotEncoder(categorical_features = categorical_feature_mask, sparse=True, handle_unknown='ignore')
# ohe.fit(X_train)
# X_train_ohe = ohe.transform(X_train)


clf_g = GradientBoostingClassifier(random_state=0, learning_rate=0.1, max_depth=5, subsample=0.5).fit(X_train, y_train)
print('Finished training the model')

####################################################################
# submit
dtypes = {  'ip': 'uint16',
            'app': 'uint16',
            'device': 'uint16',
            'os': 'uint16',
            'channel': 'uint16'}

test = pd.read_csv('../input/test.csv', dtype=dtypes, parse_dates=['click_time'])

test['click_hour'] = test['click_time'].dt.hour
X_test = test[features]
print('Finished preparing the test data')

# pred
test['is_attributed'] = clf_g.predict(X_test)
# create submission
sub_cols = ['click_id', 'is_attributed']
df_sub = test[sub_cols]
# save
df_sub.to_csv('submission_v2.csv', index=False)

# Current score on kaggle is 0.89873 
# 