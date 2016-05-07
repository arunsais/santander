

import pandas as pd
import copy
import numpy as np


def get_weighted_avg(X, weights):
    X = np.dot(np.reshape(weights, (1, X.shape[0])), X)
    X = np.average(X, axis = 0)
    return X

def ensemb(pred_file_list, weights, out_file):
    weights = weights * 1.0 / np.sum(weights)
    A = [pd.read_csv(pred_file) for pred_file in pred_file_list]
    X = [np.asarray(a.TARGET) for a in A]
    X = np.asmatrix(X)
    X = get_weighted_avg(X, weights)
    X = np.squeeze(np.asarray(X))
    ans = copy.deepcopy(A[0])
    ans.TARGET = X
    ans.to_csv(out_file, index=False)
    
def get_weights(private_scores):
    weights = (np.log(private_scores / (1-private_scores)))**1.44
    weights = weights * 1.0 / np.sum(weights)
    return weights


pred_file_list = ['./xgb_feature_selection_0.840577027088.csv', './xgb_rules.csv','./logReg_l2_cv.csv' ]
private_scores = np.asarray([0.840577027088,0.841024, 0.61508 ])

out_file = './ensemb_xgb_feature_selection__xgb_rules__logReg.csv'
weights = get_weights(private_scores)
ensemb(pred_file_list, weights, out_file)

