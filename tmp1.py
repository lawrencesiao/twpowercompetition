import pandas as pd

df= pd.read_csv('df.csv',encoding='utf8')

df_test = df.query("en_name=='MEGI' or en_name == 'MERANTIANDMALAKAS'")
df_train = df.query("en_name!='MEGI' or en_name != 'MERANTIANDMALAKAS'")


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

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(df_train.ix[:,1:], df_train.ix[:,:1], test_size=0.2)


y_train = y_train['elect_down'].values
y_test = y_test['elect_down'].values
X_train = X_train.values
X_test = X_test.values

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'auc'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)

print('Save model...')
# save model to file
gbm.save_model('model.txt')

print('Start predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# eval
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)