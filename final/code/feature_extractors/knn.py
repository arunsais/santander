import numpy as np
import pandas as pd
import sklearn as sk

from sklearn import decomposition
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__": 
	# load data
	df_train = pd.read_csv('train_art.csv')
	df_test = pd.read_csv('test_art.csv')
	df_test.ID = 0;
	y = df_train.as_matrix(columns = ['TARGET'])
	y_test = df_test.as_matrix(columns = ['ID'])
	y = np.vstack((y, y_test))
	
	
	#remove cat features
	#'var15', 'saldo_medio_var_ult1', 'var38', 'saldo_medio_v13_crt_ul1', 'delta_imp_aport_var13_1y3', 'ind_var24', 'saldo_var30'
	train_tmp = df_train[['var15', 'art_logvar38', 'saldo_medio_var5_ult3', 'saldo_medio_var5_hace3', 'num_var45_ult3', 'saldo_medio_var5_hace2', 'saldo_var30', 'num_var45_hace3', 'num_var45_hace2', 'saldo_var42',  'num_var22_ult3', 'saldo_medio_var5_ult1', 'saldo_var5', 'var38', 'n0']]
	test_tmp = df_test[['var15', 'art_logvar38', 'saldo_medio_var5_ult3', 'saldo_medio_var5_hace3', 'num_var45_ult3', 'saldo_medio_var5_hace2', 'saldo_var30', 'num_var45_hace3', 'num_var45_hace2', 'saldo_var42',  'num_var22_ult3', 'saldo_medio_var5_ult1', 'saldo_var5', 'var38', 'n0']]

	num_train = np.shape(df_train)[0]; num_test = np.shape(df_test)[0]
	X_train = pd.DataFrame.as_matrix(train_tmp)
	X_test = pd.DataFrame.as_matrix(test_tmp)

	# Scale data
	scaler = StandardScaler()
	scaler.fit(np.vstack((X_train, X_test)))
	X_train = scaler.transform(X_train)

	# apply same transformation to test data
	X_test = scaler.transform(X_test) 

	# get labels of k nearest neighbours
	nbrs = NearestNeighbors(n_neighbors=60, algorithm='kd_tree').fit(np.vstack((X_train, X_test)))

	distances, indices = nbrs.kneighbors(X_train); indices = indices[:,1:]
	X_train = np.multiply(y[indices][:,:,0], np.exp(-distances[:,1:]));
	X_train = np.vstack([np.sum(X_train[:,:5], axis = 1), np.sum(X_train[:,:15], axis = 1), np.sum(X_train[:,:30], axis = 1), np.sum(X_train, axis = 1)]).T
	
	distances2, indices2 = nbrs.kneighbors(X_test); indices2 = indices2[:,1:]
	X_test = np.multiply(y[indices2][:,:,0], np.exp(-distances2[:,1:])); 
	X_test = np.vstack([np.sum(X_test[:,:5], axis = 1), np.sum(X_test[:,:15], axis = 1), np.sum(X_test[:,:30], axis = 1), np.sum(X_test, axis = 1)]).T

	np.savez('train_knn', X_train)
	np.savez('test_knn', X_test)
