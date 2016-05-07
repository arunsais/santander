import numpy as np
import pandas as pd
import sklearn as sk

from sklearn import decomposition
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__": 
	# load data
	df_train = pd.read_csv('train.csv')
	df_test = pd.read_csv('test.csv')

	y = df_train.ix[:,-1];X = df_train.ix[:,1:-1]
	y_test = df_test.ix[:,0]; X_test = df_test.ix[:,1:]

	# add feature that counts the number of 0's
	df_train['n0'] = (X==0).sum(axis=1); df_test['n0'] = (X_test==0).sum(axis=1);
	
	# add a new feature checking if var3(country) is -999999; possibly categorical
	df_train['art_var3'] = [1 if i == -999999 else 0 for i in df_train['var3']] 
	df_test['art_var3'] = [1 if i == -999999 else 0 for i in df_test['var3']] 

	#var36 possibly categorical; possible values [0, 1, 2, 3, 99]
	df_train['art_var36'] = [1 if i == 99 else 0 for i in df_train['var36']] 
	df_test['art_var36'] = [1 if i == 99 else 0 for i in df_test['var36']]
	
	# var38mc == 1 when var38 has the most common value and 0 otherwise
	# logvar38 is log transformed feature when var38mc is 0, zero otherwise
	df_train['art_var38mc'] = np.isclose(df_train.var38, 117310.979016)
	df_train['art_logvar38'] = df_train.loc[~df_train['art_var38mc'], 'var38'].map(np.log)
	df_train.loc[df_train['art_var38mc'], 'art_logvar38'] = 0
	df_train['art_var38mc'] = [1 if i else 0 for i in df_train['art_var38mc']]

	df_test['art_var38mc'] = np.isclose(df_test.var38, 117310.979016)
	df_test['art_logvar38'] = df_test.loc[~df_test['art_var38mc'], 'var38'].map(np.log)
	df_test.loc[df_test['art_var38mc'], 'art_logvar38'] = 0
	df_test['art_var38mc'] = [1 if i else 0 for i in df_test['art_var38mc']]

	# other possible categorical features: num_var13_corto, num_var13_corto_0, num_var24_0, num_var12, num_var5, num_var5_0, num_var12_0, num_var13, num_var13_0, num_var42, num_var4, num_var42_0, num_var30, num_var39_0, num_var41_0

	# PCA
	# only on important features
	train_tmp = df_train[['var15', 'art_logvar38', 'saldo_medio_var5_ult3', 'saldo_medio_var5_hace3', 'num_var45_ult3', 'saldo_medio_var5_hace2', 'saldo_var30', 'num_var45_hace3', 'num_var45_hace2', 'saldo_var42',  'num_var22_ult3', 'saldo_medio_var5_ult1', 'saldo_var5', 'var38', 'n0', 'saldo_medio_var8_ult1', 'saldo_medio_var12_ult1', 'saldo_medio_var33_ult1', 'delta_imp_aport_var13_1y3', 'ind_var24', 'saldo_medio_var13_corto_ult1', 'num_var22_hace3', 'num_var22_hace2', 'num_var45_ult1', 'num_var22_ult1']]
	test_tmp = df_test[['var15', 'art_logvar38', 'saldo_medio_var5_ult3', 'saldo_medio_var5_hace3', 'num_var45_ult3', 'saldo_medio_var5_hace2', 'saldo_var30', 'num_var45_hace3', 'num_var45_hace2', 'saldo_var42',  'num_var22_ult3', 'saldo_medio_var5_ult1', 'saldo_var5', 'var38', 'n0', 'saldo_medio_var8_ult1', 'saldo_medio_var12_ult1', 'saldo_medio_var33_ult1', 'delta_imp_aport_var13_1y3', 'ind_var24', 'saldo_medio_var13_corto_ult1', 'num_var22_hace3', 'num_var22_hace2', 'num_var45_ult1', 'num_var22_ult1']]
	df = pd.concat([train_tmp, test_tmp]);
	num_train = np.shape(df_train)[0]; num_test = np.shape(df_test)[0]
	data = pd.DataFrame.as_matrix(df)

	# Scale data
	scaler = StandardScaler()
	scaler.fit(data)
	data = scaler.transform(data)
	
	pca = decomposition.PCA(n_components=10, copy = False, whiten = False)
	data = pca.fit(data).transform(data)
	for i in range(np.shape(data)[1]):
		df_train['pca'+str(i)] = data[:num_train,i]
		df_test['pca'+str(i)] = data[num_train:,i]

	# PCA, all features
	train_tmp = df_train.drop(['num_var13_corto', 'num_var13_corto_0', 'num_var24_0', 'num_var12', 'num_var5', 'num_var5_0', 'num_var12_0', 'num_var13', 'num_var13_0', 'num_var42', 'num_var4', 'num_var42_0', 'num_var30', 'num_var39_0', 'num_var41_0', 'var3', 'ID', 'TARGET', 'pca0', 'pca1', 'pca2', 'pca3', 'pca4', 'pca5', 'pca6', 'pca7', 'pca8', 'pca9'], 1)
	test_tmp = df_test.drop(['num_var13_corto', 'num_var13_corto_0', 'num_var24_0', 'num_var12', 'num_var5', 'num_var5_0', 'num_var12_0', 'num_var13', 'num_var13_0', 'num_var42', 'num_var4', 'num_var42_0', 'num_var30', 'num_var39_0', 'num_var41_0', 'var3', 'ID', 'pca0', 'pca1', 'pca2', 'pca3', 'pca4', 'pca5', 'pca6', 'pca7', 'pca8', 'pca9'], 1)
	df = pd.concat([train_tmp, test_tmp]);
	num_train = np.shape(df_train)[0]; num_test = np.shape(df_test)[0]
	data = pd.DataFrame.as_matrix(df)

	# Scale data
	scaler = StandardScaler()
	scaler.fit(data)
	data = scaler.transform(data)
	
	pca = decomposition.PCA(n_components=10, copy = False, whiten = False)
	data = pca.fit(data).transform(data)
	for i in range(np.shape(data)[1]):
		df_train['pca_all'+str(i)] = data[:num_train,i]
		df_test['pca_all'+str(i)] = data[num_train:,i]
	

	#make sure TARGET column is at the end
	cols = df_train.drop(['TARGET'], 1).columns.tolist()
	cols = cols + ['TARGET']
	df_train[cols].to_csv('train_art.csv', index=False)	
	df_test.to_csv('test_art.csv', index=False)
