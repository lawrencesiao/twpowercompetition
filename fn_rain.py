from datetime import datetime, timedelta
import pandas as pd

rain = pd.read_csv('tp_rain.csv')
rain['ts'] = [datetime.strptime(i,'%Y/%m/%d %H:%M') for i in rain.accu_end_time]
ref_df = rain

def find_pre(way,stno,ts,ref_df,pre):
	step = timedelta(seconds=3600)
	result=[]
	print(ts)
	print(stno)
	for i in range(pre):
		value = ref_df.loc[(ref_df['stno'] == stno) & (ref_df['ts'] == ts)]['accu_value'].values
		if value.any():
			result.append(value[0])
		else:
			result.append(0)
		ts -= step
	print(pre)
	print(result)
	if way=='max':
		return max(result)
	elif way=='mean':
		return reduce(lambda x, y: x + y, result) / len(result)
	else:
		return sum(result)

rain['pre_accu3'] = rain.apply(lambda row: find_pre('sum',row['stno'], row['ts'],ref_df,pre=3), axis=1)
rain['pre_accu6'] = rain.apply(lambda row: find_pre('sum',row['stno'], row['ts'],ref_df,pre=6), axis=1)
rain['pre_accu12'] = rain.apply(lambda row: find_pre('sum',row['stno'], row['ts'],ref_df,pre=12), axis=1)

rain['pre_max3'] = rain.apply(lambda row: find_pre('max',row['stno'], row['ts'],ref_df,pre=3), axis=1)
rain['pre_max6'] = rain.apply(lambda row: find_pre('max',row['stno'], row['ts'],ref_df,pre=6), axis=1)
rain['pre_max12'] = rain.apply(lambda row: find_pre('max',row['stno'], row['ts'],ref_df,pre=12), axis=1)

rain['pre_mean3'] = rain.apply(lambda row: find_pre('mean',row['stno'], row['ts'],ref_df,pre=3), axis=1)
rain['pre_mean6'] = rain.apply(lambda row: find_pre('mean',row['stno'], row['ts'],ref_df,pre=6), axis=1)
rain['pre_mean12'] = rain.apply(lambda row: find_pre('mean',row['stno'], row['ts'],ref_df,pre=12), axis=1)



