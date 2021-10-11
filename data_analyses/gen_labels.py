import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/Yuchen/Downloads/Batch_3851507_batch_results.csv')
#df=pd.read_csv('test.csv')
print("len:",len(df))

#df = df[df['Answer.sentiment'].str.isnumeric()]
df=df[pd.to_numeric(df['Answer.sentiment'], errors='coerce').notnull()]
#df=df[df[['Answer.sentiment']].apply(lambda x: x[0].isdigit(), axis=1)]
#df=df[df['Answer.sentiment'].apply(lambda x: isinstance(x, (int, float)))]
#df[df['Answer.sentiment'].apply(lambda x: type(x) in [int, np.int64, float, np.float64])]
#print(type(df['Answer.sentiment'][1710]) in [int, np.int64, float, np.float64])
#print(df)
print("len:",len(df))

new_df=pd.DataFrame({'dir':[],'sentiment':[]})
#new_df['dir']=df['Input.VIDEO_ID'].str.slice(41,)
new_df['dir']=df['Input.VIDEO_ID'].str.slice(41,).replace(':','_', regex=True)
#new_df['dir']=df['Input.VIDEO_ID'].apply(labda x: x)
new_df['sentiment']=df['Answer.sentiment'].astype('float32')
#print(len(new_df))

# new_df=pd.DataFrame({'dir': ['A','A','A','B','B','B','B'], 'sentiment': [1.1,11.2,1.1,3.3,3.40,3.3,100.0]})
# print(new_df)
def is_outlier(s):
    lower_limit = s.mean() - (s.std() * 3)
    upper_limit = s.mean() + (s.std() * 3)
    return ~s.between(lower_limit, upper_limit)

#print(new_df)
new_df=new_df.groupby('dir').mean()
#new_df = new_df[~new_df.groupby('dir')['sentiment'].apply(is_outlier)].groupby('dir').median()
print("len:",len(new_df))
#print(new_df)

new_df.to_csv("sentiment_label_french.csv", encoding='utf-8')


# calculate how many mp4 files are there in total
base_dir='C:/Users/Yuchen/Downloads/french_amt_b1_b2/zipper_french/'

import os
path=os.getcwd()
count=0
for root, dir, files in os.walk(base_dir):
	for each in files:
		ext=os.path.splitext(each)[1]
		if ext=='.mp4':
			count+=1
print("count:",count)









#new_df.groupby('dir')
#print(new_df)
#new_df['dir']=new_df['dir'][41:]


# dict_temp=dict()
# for row in df:
# 	print(row)
# 	if row['Input.VIDEO_ID'][:41]=="/cache/multilingual/french/data/FR_B1_B2/":
# 			key=row['Input.VIDEO_ID'][41:]
# 	else:
# 		print("error!")
# 	value=int(row['Answer.sentiment'])
# 	dict_temp[key].append(value)




# import csv

# with open('C:/Users/Yuchen/Downloads/Batch_3851507_batch_results.csv', newline='') as csvfile:
# 	spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
# 	for row in spamreader:
# 		 print(', '.join(row))
# 		 break