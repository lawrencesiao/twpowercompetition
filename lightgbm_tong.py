from sklearn.model_selection import KFold
# coding: utf-8
# pylint: disable = invalid-name, C0111
# ssh -L 8000:localhost:8888 your_server_username@your_server_ip
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error,make_scorer
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn import preprocessing

mean_squared_error_ = make_scorer(mean_squared_error,greater_is_better=False)

df_raw= pd.read_csv('df_1107v2.csv',encoding='utf8')

df= pd.read_csv('df_1107v2.csv',encoding='utf8')


test_idx = df.loc[(df.typhoon=='NESATANDHAITANG')|(df.typhoon=='MEGI'),:].index
train_idx = df.loc[(df.typhoon!='NESATANDHAITANG') & (df.typhoon!='MEGI'),:].index

def preprocessor(df):
	res = ['elect_down']

	cate_fs =['year','type','alert_level','arrive_weekday','arrive_week','double_kill'\
	,'region_cluster','landslide_alert']

	cont_fs = ['magnitude','hpa','wind_speed','r7_km','r10_km','duration_h','pole_type_counts',
	     'p1', 'p2', 'p3','p4','p5','p6','p7','p8','p9', 'p10',
	          'pole_counts','people_total','area','population_density',
	               'mean_accu_hour_rain', 'mean_accu_day_rain', 'accu_rain',
	           'heavy_rain_count_rule1', 'heavy_rain_count_rule2',
	           'how_rain_count_rule1', 'big_how_rain_count',
	          'big_big_how_rain_count','mean_hr_wsmax', 'mean_hr_wsgust', 'max_hr_wsmax', 'max_hr_wsgust']

	df = df[res+ cate_fs + cont_fs]
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
                        num_leaves=40,
                        learning_rate=0.5,
                        n_estimators=20,
                        lambda_l2=0.24
                        )
gbm.fit(X_train, y_train,
        eval_metric='l2')

y_pred = gbm.predict(X_test)

y_pred[y_pred<0] = 0
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

result_f.to_csv('submit_lgbm_l2_024.csv',encoding='utf8', index=False)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(gbm, X_train, y_train, cv=5,scoring='mean_squared_error')
print(sum(scores)/len(scores))



