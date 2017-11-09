from sklearn.model_selection import KFold
# coding: utf-8
# pylint: disable = invalid-name, C0111
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df_raw= pd.read_csv('df4.csv',encoding='utf8')

df= pd.read_csv('df4.csv',encoding='utf8')

#df['type'][[i!=2 for i in df['type']]] = 0
#df['type'][[i==2 for i in df['type']]] = 1

#df['stname'][df.stname != u'梧棲'] = 0
#df['stname'][df.stname == u'梧棲'] = 1

def preprocessor(df):
	res = ['elect_down']
	cate_fs = ['year','type','magnitude','arrive_month','arrive_hour','arrive_weekday','arrive_week','stname',\
	'double_kill']

	cont_fs = ['r7_km','p2','p5','p6',\
	'p7','pole_counts','people_total','area','population_density']

	cont_fs2 =['mean_accu_hour_rain','heavy_rain_count_rule2' , 'max_hour_accu_rain', 'max_day_accu_rain', 'mean_hr_wsmax', 'mean_hr_wsgust', 'max_hr_wsmax',	'max_hr_wsgust' ,'region_cluster']

	df = df[res+ cate_fs + cont_fs + cont_fs2 + ['en_name']]
	for f in cate_fs:
		one_hot = pd.get_dummies(df[f],prefix=f)
		df = df.drop(f, axis=1)
		df = df.join(one_hot)
	return df

df = preprocessor(df)

df.ix[:,1:] = preprocessing.normalize(df.ix[:,1:].values, norm='l2')

df_raw = df_raw.query("en_name=='MEGI' or en_name == 'NESATANDHAITANG'")

df_test = df.query("en_name=='MEGI' or en_name == 'NESATANDHAITANG'")
df_train = df.query("en_name!='MEGI' and en_name != 'NESATANDHAITANG'")

df_test = df_test.drop('en_name', axis=1)
df_train = df_train.drop('en_name', axis=1)



y_train = df_train['elect_down'].values
y_test = df_test['elect_down'].values

X_train = preprocessing.normalize(df_train.ix[:,1:].values, norm='l2')

X_test = preprocessing.normalize(df_test.ix[:,1:].values, norm='l2')

gbm = lgb.LGBMRegressor(objective='regression',
                        num_leaves=5,
                        learning_rate=0.05,
                        n_estimators=3)
gbm.fit(X_train, y_train,
        eval_metric='l2')


#feature_importances = pd.DataFrame(df_train.ix[:,1:].columns,gbm.feature_importances_)
#feature_importances.to_csv('/Users/lawrencesiao/twpowercompetition/FI.csv',encoding='utf8')

print('Start predicting...')
# predict
y_pred = gbm.predict(X_test)
# eval


##
df_raw = df_raw[]
df_c = pd.concat([df_raw.ix[:,:5].reset_index(drop=True), pd.DataFrame(y_pred,columns=['pred'])], axis=1)

df_MEGI = df_c.query("typhoon=='MEGI'")
df_NESATANDHAITANG = df_c.query("typhoon=='NESATANDHAITANG'")

submit = pd.read_csv('submit.csv',encoding='utf8')

result = submit.merge(df_NESATANDHAITANG, left_on=['VilCode'], right_on=['VilCode'], how='inner')

result = result.merge(df_MEGI, left_on=['VilCode'], right_on=['VilCode'], how='inner')

result_f = result[['CityName_x','TownName_x','VilCode','VilName_x','pred_x','pred_y']]
result_f.columns = ['CityName','TownName','VilCode','VilName','NesatAndHaitang','NesatAndHaitang']

result_f.to_csv('submit_lgbm.csv',encoding='utf8', index=False)

# feature importances
qq = pd.DataFrame(df_train.ix[:,1:].columns,gbm.feature_importances_)
qq.to_csv('/Users/lawrencesiao/twpowercompetition/FI.csv',encoding='utf8')

estimator = lgb.LGBMRegressor()

param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'n_estimators': [10,20,30, 40,50,60,70],
    'num_leaves': [10,20,30,40,50]
}

gbm = GridSearchCV(estimator, param_grid)

gbm.fit(X_train, y_train)

print('Best parameters found by grid search are:', gbm.best_params_)