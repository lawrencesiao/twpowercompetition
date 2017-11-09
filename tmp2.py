import pandas as pd

df= pd.read_csv('df2.csv',encoding='utf8')

df_test = df.query("en_name=='MEGI' or en_name == 'MERANTIANDMALAKAS'")
df_train = df.query("en_name!='MEGI' or en_name != 'MERANTIANDMALAKAS'")


def preprocessor(df):
	res = ['elect_down']
	cate_fs = ['year','type','magnitude','arrive_month','arrive_hour','arrive_weekday','arrive_week','stname',\
	'double_kill']

	cont_fs = ['hpa','wind_speed','r7_km','r10_km','alert_level','duration_h','p1','p2','p3','p4','p5','p6',\
	'p7','p8','p9','p10','pole_counts','people_total','area','population_density']

	df = df[res+ cate_fs + cont_fs]
	for f in cate_fs:
		one_hot = pd.get_dummies(df[f],prefix=f)
		df = df.drop(f, axis=1)
		df = df.join(one_hot)
	return df

df_test = preprocessor(df_test)
df_train = preprocessor(df_train)

y_train = df_train['elect_down'].values
y_test = df_test['elect_down'].values
X_train = df_train.ix[:,1:].values
X_test = df_test.ix[:,1:].values


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

