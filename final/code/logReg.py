import numpy as np
import pandas as pd
import sklearn as sk

from itertools import combinations
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import OneHotEncoder

# load data
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

y = np.asarray(df_train.ix[:,-1])
X = np.asarray(df_train.ix[:,1:-1])
test_ID = np.asarray(df_test.ix[:,0])
X_test = np.asarray(df_test.ix[:,1:])

# CV and training
classifier = LogisticRegressionCV(penalty='l2', solver = 'liblinear', verbose = 100, n_jobs = 4, max_iter = 10000, scoring = 'roc_auc')
classifier.fit(X, y)

print(classifier.scores_)

# testing
y_test_pred = classifier.predict_proba(X_test)
ans = pd.DataFrame({'ID': test_ID, 'TARGET' : y_test_pred[:,1]})
ans.to_csv('logReg_l2_cv.csv', index=False, columns=['ID', 'TARGET'])