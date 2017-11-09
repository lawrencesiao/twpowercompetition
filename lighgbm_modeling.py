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

df['type'][[i!=2 and i!=7 for i in df['type']]] = 0

df['arrive_month'][[i!=8 for i in df['arrive_month']]] = 0

df['stname'][[i not in [u'梧棲',u'淡水',u'臺中',u'蘇澳',u'鞍部', u'高雄'] for i in df['stname']]] = 'other'


test_idx = df.loc[(df.typhoon=='NESATANDHAITANG')|(df.typhoon=='MEGI'),:].index
train_idx = df.loc[(df.typhoon!='NESATANDHAITANG') & (df.typhoon!='MEGI'),:].index

def preprocessor(df):
	res = ['elect_down']
	cate_fs = ['type','magnitude','stname',\
	'double_kill']

	cont_fs = ['hpa','wind_speed','r10_km','duration_h','p2','p5','p6',\
	'p7','pole_counts','people_total','area','population_density']

	cont_fs2 =['mean_accu_hour_rain','heavy_rain_count_rule2','how_rain_count_rule1'
	, 'max_hr_wsmax','max_hr_wsgust' ,'region_cluster']
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
                        num_leaves=105,
                        learning_rate=1.5,
                        n_estimators=2)
gbm.fit(X_train, y_train,
        eval_metric='l2')


#feature_importances = pd.DataFrame(df_train.ix[:,1:].columns,gbm.feature_importances_)
#feature_importances.to_csv('/Users/lawrencesiao/twpowercompetition/FI.csv',encoding='utf8')

print('Start predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
# eval
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)


##
df_raw_test =df_raw.loc[[i for i in test_idx]]

df_c = pd.concat([df_raw_test.ix[:,:5].reset_index(drop=True), pd.DataFrame(y_pred,columns=['pred'])], axis=1)

df_MEGI = df_c.query("typhoon=='MEGI'")
df_NESATANDHAITANG = df_c.query("typhoon=='NESATANDHAITANG'")

submit = pd.read_csv('submit.csv',encoding='utf8')

result = submit.merge(df_NESATANDHAITANG, left_on=['VilCode'], right_on=['VilCode'], how='inner')

result = result.merge(df_MEGI, left_on=['VilCode'], right_on=['VilCode'], how='inner')

result_f = result[['CityName_x','TownName_x','VilCode','VilName_x','pred_x','pred_y']]
result_f.columns = ['CityName','TownName','VilCode','VilName','NesatAndHaitang','Megi']

result_f.to_csv('submit_lgbm2.csv',encoding='utf8', index=False)

# feature importances
print('Feature importances:', list(gbm.feature_importances_))
qq = pd.DataFrame(df_train.ix[:,1:].columns,gbm.feature_importances_)
qq.to_csv('/Users/lawrencesiao/twpowercompetition/FI.csv',encoding='utf8')
estimator = lgb.LGBMRegressor()

param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'n_estimators': [10,20,30],
    'num_leaves': [40,50,60,70,80]
}

gbm = GridSearchCV(estimator, param_grid,cv=5)

gbm.fit(X_train, y_train)

print('Best parameters found by grid search are:', gbm.best_params_)