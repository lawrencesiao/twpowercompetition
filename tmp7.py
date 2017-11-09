from sklearn.model_selection import KFold
# coding: utf-8
# pylint: disable = invalid-name, C0111
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn import preprocessing
df_raw= pd.read_csv('df4.csv',encoding='utf8')

df= pd.read_csv('df4.csv',encoding='utf8')

#df['type'][[i!=2 for i in df['type']]] = 0
#df['type'][[i==2 for i in df['type']]] = 1

#df['stname'][df.stname != u'梧棲'] = 0
#df['stname'][df.stname == u'梧棲'] = 1

test_idx = df.loc[(df.typhoon=='NESATANDHAITANG')|(df.typhoon=='MEGI'),:].index
train_idx = df.loc[(df.typhoon!='NESATANDHAITANG') & (df.typhoon!='MEGI'),:].index

def preprocessor(df):
	res = ['elect_down']
	cate_fs = ['year','type','magnitude','arrive_month','arrive_hour','arrive_weekday','arrive_week','stname',\
	'double_kill']

	cont_fs = ['hpa','wind_speed','r7_km','r10_km','alert_level','duration_h','p1','p2','p3','p4','p5','p6',\
	'p7','p8','p9','p10','pole_counts','people_total','area','population_density']

	cont_fs2 =['mean_accu_hour_rain', 'mean_accu_day_rain','accu_rain','heavy_rain_count_rule1','heavy_rain_count_rule2','how_rain_count_rule1', 'big_how_rain_count' ,'big_big_how_rain_count', 'max_hour_accu_rain', 'max_day_accu_rain', 'mean_hr_wsmax', 'mean_hr_wsgust', 'max_hr_wsmax',	'max_hr_wsgust' ,'region_cluster']

	df = df[res+ cate_fs + cont_fs + cont_fs2]
	for f in cate_fs:
		one_hot = pd.get_dummies(df[f],prefix=f)
		df = df.drop(f, axis=1)
		df = df.join(one_hot)
	return df

df = preprocessor(df)
df.ix[:,1:] = preprocessing.normalize(df.ix[:,1:].values, norm='l2')

df_test = df.loc[[i for i in test_idx]]
df_train = df.loc[[i for i in train_idx]]


X_train = df_train.ix[:,1:].values
X_test = df_test.ix[:,1:].values


y_train = df.loc[[i for i in train_idx],'elect_down'].values
y_test = df.loc[[i for i in test_idx],'elect_down'].values

gbm = lgb.LGBMRegressor(objective='regression',
                        num_leaves=31,
                        learning_rate=0.05,
                        n_estimators=20)
gbm.fit(X_train, y_train,
        eval_metric='l2')


print('Feature importances:', list(gbm.feature_importances_))
qq = pd.DataFrame(df_train.ix[:,1:].columns,gbm.feature_importances_)
qq.to_csv('/Users/lawrencesiao/twpowercompetition/FI.csv',encoding='utf8')
estimator = lgb.LGBMRegressor()
