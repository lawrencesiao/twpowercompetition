from sklearn.model_selection import KFold
# coding: utf-8
# pylint: disable = invalid-name, C0111
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import pandas as pd

df= pd.read_csv('df4.csv',encoding='utf8')

df['type'][[i!=2 for i in df['type']]] = 0
df['type'][[i==2 for i in df['type']]] = 1

df['stname'][df.stname != u'梧棲'] = 0
df['stname'][df.stname == u'梧棲'] = 1

def preprocessor(df):
	res = ['elect_down']
	cate_fs = ['type','stname']

	cont_fs = ['r7_km','p2','p5','p6',\
	'p7','pole_counts','people_total','area','population_density']

	cont_fs2 =['mean_accu_hour_rain','heavy_rain_count_rule2' , 'max_hour_accu_rain', 'max_day_accu_rain', 'mean_hr_wsmax', 'mean_hr_wsgust', 'max_hr_wsmax',	'max_hr_wsgust' ,'region_cluster']

	df = df[res+ cate_fs + cont_fs + cont_fs2]
	for f in cate_fs:
		one_hot = pd.get_dummies(df[f],prefix=f)
		df = df.drop(f, axis=1)
		df = df.join(one_hot)
	return df


df_test = df.query("en_name=='MEGI' or en_name == 'NESATANDHAITANG'")
df_train = df.query("en_name!='MEGI' and en_name != 'NESATANDHAITANG'")

df_test = preprocessor(df_test)
df_train = preprocessor(df_train)

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(df_train.ix[:,1:], df_train.ix[:,:1], test_size=0.2)


y_train = y_train['elect_down'].values
y_test = y_test['elect_down'].values
X_train = X_train.values
X_test = X_test.values

gbm = lgb.LGBMRegressor(objective='regression',
                        num_leaves=31,
                        learning_rate=0.05,
                        n_estimators=20)
gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='l2',
        early_stopping_rounds=5)

qq = pd.DataFrame(df_train.ix[:,1:].columns,gbm.feature_importances_)
qq.to_csv('/Users/lawrencesiao/twpowercompetition/FI.csv',encoding='utf8')

print('Start predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
# eval
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

# feature importances
print('Feature importances:', list(gbm.feature_importances_))

estimator = lgb.LGBMRegressor()

param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'n_estimators': [10,20,30, 40,50,60,70],
    'num_leaves': [10,20,30,40,50]
}

gbm = GridSearchCV(estimator, param_grid)

gbm.fit(X_train, y_train,eval_metric='l2')

print('Best parameters found by grid search are:', gbm.best_params_)