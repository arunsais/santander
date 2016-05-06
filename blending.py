
import numpy as np
import pandas as pd
import sklearn as sk
import xgboost as xgb

from scipy import sparse
from itertools import combinations
from scipy.sparse import coo_matrix, hstack
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.grid_search import GridSearchCV
from sklearn.cluster import KMeans
from  sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
	# load data
	df_train = pd.read_csv('../train_art_f.csv')
	df_test = pd.read_csv('../test_art_f.csv')

	# remove columns with 0 variance
	remove = []
	cols = df_train.columns
	for i in range(len(cols)-1):
		if df_train[cols[i]].std() == 0:
			remove.append(cols[i])

	# remove duplicated columns
	for i in range(len(cols)-1):
		v = df_train[cols[i]].values
		for j in range(i+1,len(cols)):
			if np.array_equal(v,df_train[cols[j]].values):
				remove.append(cols[j])	

	df_train.drop(remove, axis=1, inplace=True)
	df_test.drop(remove, axis=1, inplace=True)

	tokeep = ['num_var39_0',  # 0.00031104199066874026
			  'ind_var13',  # 0.00031104199066874026
			  'num_op_var41_comer_ult3',  # 0.00031104199066874026
			  'num_var43_recib_ult1',  # 0.00031104199066874026
			  'imp_op_var41_comer_ult3',  # 0.00031104199066874026
			  'num_var8',  # 0.00031104199066874026
			  'num_var42',  # 0.00031104199066874026
			  'num_var30',  # 0.00031104199066874026
			  'saldo_var8',  # 0.00031104199066874026
			  'num_op_var39_efect_ult3',  # 0.00031104199066874026
			  'num_op_var39_comer_ult3',  # 0.00031104199066874026
			  'num_var41_0',  # 0.0006220839813374805
			  'num_op_var39_ult3',  # 0.0006220839813374805
			  'saldo_var13',  # 0.0009331259720062209
			  'num_var30_0',  # 0.0009331259720062209
			  'ind_var37_cte',  # 0.0009331259720062209
			  'ind_var39_0',  # 0.001244167962674961
			  'num_var5',  # 0.0015552099533437014
			  'ind_var10_ult1',  # 0.0015552099533437014
			  'num_op_var39_hace2',  # 0.0018662519440124418
			  'num_var22_hace2',  # 0.0018662519440124418
			  'num_var35',  # 0.0018662519440124418
			  'ind_var30',  # 0.0018662519440124418
			  'num_med_var22_ult3',  # 0.002177293934681182
			  'imp_op_var41_efect_ult1',  # 0.002488335925349922
			  'var36',  # 0.0027993779160186624
			  'num_med_var45_ult3',  # 0.003110419906687403
			  'imp_op_var39_ult1',  # 0.0037325038880248835
			  'imp_op_var39_comer_ult3',  # 0.0037325038880248835
			  'imp_trans_var37_ult1',  # 0.004043545878693624
			  'num_var5_0',  # 0.004043545878693624
			  'num_var45_ult1',  # 0.004665629860031105
			  'ind_var41_0',  # 0.0052877138413685845
			  'imp_op_var41_ult1',  # 0.0052877138413685845
			  'num_var8_0',  # 0.005598755832037325
			  'imp_op_var41_efect_ult3',  # 0.007153965785381027
			  'num_op_var41_ult3',  # 0.007153965785381027
			  'num_var22_hace3',  # 0.008087091757387248
			  'num_var4',  # 0.008087091757387248
			  'imp_op_var39_comer_ult1',  # 0.008398133748055987
			  'num_var45_ult3',  # 0.008709175738724729
			  'ind_var5',  # 0.009953343701399688
			  'imp_op_var39_efect_ult3',  # 0.009953343701399688
			  'num_meses_var5_ult3',  # 0.009953343701399688
			  'saldo_var42',  # 0.01181959564541213
			  'imp_op_var39_efect_ult1',  # 0.013374805598755831
			  'num_var45_hace2',  # 0.014618973561430793
			  'num_var22_ult1',  # 0.017107309486780714
			  'saldo_medio_var5_ult1',  # 0.017418351477449457
			  'saldo_var5',  # 0.0208398133748056
			  'ind_var8_0',  # 0.021150855365474338
			  'ind_var5_0',  # 0.02177293934681182
			  'num_meses_var39_vig_ult3',  # 0.024572317262830483
			  'saldo_medio_var5_ult3',  # 0.024883359253499222
			  'num_var45_hace3',  # 0.026749611197511663
			  'num_var22_ult3',  # 0.03452566096423017
			  'saldo_medio_var5_hace3',  # 0.04074650077760498
			  'saldo_medio_var5_hace2',  # 0.04292379471228616
			  'n0',  # 0.04696734059097978
			  'saldo_var30',  # 0.09611197511664074
			  'var38',  # 0.1390357698289269
			  'var15', 'art_var3', 'art_var36', 'art_var38mc', 'art_logvar38','pca0', 'pca1', 'pca2', 'pca3', 'pca4', 'pca5', 'pca6', 'pca7', 'pca8', 'pca9', 'pca_all0', 'pca_all1', 'pca_all2', 'pca_all3', 'pca_all4', 'pca_all5', 'pca_all6', 'pca_all7', 'pca_all8', 'pca_all9', 'num_meses_var5_ult3+pca_all1', 'num_var30*var15', 'pca2-saldo_var24', 'pca2-pca_all6', 'num_var37_med_ult2+pca_all0'] 

	#df_train.replace(0, np.nan, True).to_sparse()
	#df_test.replace(0, np.nan, True).to_sparse()
	#0 if np.isnan(i) else 1
	y = df_train.TARGET; X = pd.DataFrame.as_matrix(df_train[tokeep]); y_id = df_train.ID;
	y_test = df_test.ID; X_test = pd.DataFrame.as_matrix(df_test[tokeep])
	del df_train, df_test

	# get knn features
	X_knn = np.load('../train_knn.npz')['arr_0']; X_test_knn = np.load('../test_knn.npz')['arr_0']
	X_knn = X_knn[:,[3]]; X_test_knn = X_test_knn[:,[3]];
	X_bnn = np.load('../train_bnn.npz')['arr_0']; X_test_bnn = np.load('../test_bnn.npz')['arr_0']
	X_bnn = X_bnn[:,[3]]; X_test_bnn = X_test_bnn[:,[3]];

	scaler = StandardScaler()
	scaler.fit(np.vstack((X, X_test)))
	X = scaler.transform(X)
	X_test = scaler.transform(X_test) 
	del scaler

	# cluster
	km = KMeans(n_clusters = 5)
	km.fit(np.vstack((X, X_test)))
	X_clusters = km.predict(X)
	X_test_clusters = km.predict(X_test)
	del km

	X = np.hstack([X_clusters.reshape(-1,1),  X])
	X_test = np.hstack([X_test_clusters.reshape(-1,1), X_test])

	clfs = [RandomForestClassifier(n_estimators=350, n_jobs=4, max_depth=6, criterion='gini',class_weight = 'balanced'),
			RandomForestClassifier(n_estimators=350, n_jobs=4, max_depth=6, criterion='entropy',class_weight = 'balanced'),
			ExtraTreesClassifier(n_estimators=1000, n_jobs=4, max_depth=6, criterion='gini',class_weight = 'balanced'),
			ExtraTreesClassifier(n_estimators=1000, n_jobs=4, max_depth=6, criterion='entropy',class_weight = 'balanced'),
			xgb.XGBClassifier(learning_rate=0.02, subsample=0.68, nthread=4, colsample_bytree=0.7, max_depth=5, n_estimators=560, objective= 'binary:logistic'),
			LogisticRegression(C = 0.3535, penalty='l1', n_jobs = 3, solver = 'liblinear')]

	skf = list(StratifiedKFold(y, 10))
	print "Creating train and test sets for blending."
    
	dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
	dataset_blend_test = np.zeros((X_test.shape[0], len(clfs)))

	for j, clf in enumerate(clfs):
		print j, clf
		dataset_blend_test_j = np.zeros((X_test.shape[0], len(skf)))
		for i, (train, test) in enumerate(skf):
			print "Fold", i
			X_train = X[train]
			y_train = y[train]
			X_eval = X[test]
			y_eval = y[test]
			clf.fit(X_train, y_train)
			y_eval = clf.predict_proba(X_eval)[:,1]
			dataset_blend_train[test, j] = y_eval
			dataset_blend_test_j[:, i] = clf.predict_proba(X_test)[:,1]
		dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)

	np.savez('dataset_blend_test', dataset_blend_test);	np.savez('dataset_blend_train', dataset_blend_train)
	X = np.hstack([dataset_blend_train,  X_knn])
	X_test = np.hstack([dataset_blend_test,  X_test_knn])

	print "Blending."
	grid = { 'max_depth':[3, 5], 'n_estimators':[50], 'learning_rate':[0.01, 0.02, 0.03], 'subsample':[0.8]}
	grid_search = GridSearchCV(estimator = xgb.XGBClassifier(missing=np.nan, nthread = 4, objective= 'binary:logistic', seed=9), param_grid = grid, scoring='roc_auc', cv=5)
	grid_search.fit(X, y)
	print grid_search.grid_scores_, grid_search.best_score_		

	# train with best parameters
	clf = xgb.XGBClassifier(nthread = 4, n_estimators=grid_search.best_params_['n_estimators'], max_depth=grid_search.best_params_['max_depth'], objective= 'binary:logistic',  learning_rate = grid_search.best_params_['learning_rate'], subsample = grid_search.best_params_['subsample'], seed = 9)
	clf.fit(X, y)

	# testing
	y_test_pred = clf.predict_proba(X_test)
	ans = pd.DataFrame({'ID': y_test, 'TARGET' : y_test_pred[:,1]})
	ans.to_csv('blending_xgb_test_lr.csv', index=False, columns=['ID', 'TARGET'])
    
	y_pred = clf.predict_proba(X)
	ans = pd.DataFrame({'ID': y_id, 'TARGET' : y_pred[:,1]})
	ans.to_csv('blending_xgb_train_lr.csv', index=False, columns=['ID', 'TARGET'])
