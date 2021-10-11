import pickle
import csv
import torch
import os
import sys
from torch.utils.data import Dataset, DataLoader
import numpy as np

class build_mit_swav(Dataset):
	def __init__(self, file_dir, transform=None):

		data_list=list()

		raw_data=pickle.load(open(file_dir,"rb"))
		embeddings=raw_data['embeds']
		words=raw_data['words']
		for i,word in enumerate(words):
			for embeds in embeddings[word]['visual']:
				data_list.append((embeds,i))

		self.data_list=data_list

	def __len__(self):
		return len(self.data_list)

	def __getitem__(self, idx):
		if isinstance(idx, list):
			# print("list!")
			return [self.data_list[i] for i in idx]
		else:
			# print("integer!")
			return self.data_list[idx]

class build_moseas(Dataset):
	def __init__(self):
		self.data=list()
		self.get_labels()
		self.get_features()
		print(len(self.label_dict))
		print(len(self.feature_dict))
		intersection_keys=list(set(list(self.label_dict.keys()))&set(list(self.feature_dict.keys())))
		#print(intersection_keys)
		#print(len(intersection_keys))
		for key in intersection_keys:
			self.data.append((self.feature_dict[key],self.label_dict[key]))

	def __len__(self):
		return len(self.data)
	def __getitem__(self, idx):
		return self.data[idx]

	def get_labels(self):
		self.label_dict=dict()
		label_record=dict()
		with open('sentiment_label_french.csv', newline='') as csvfile:
			reader = csv.reader(csvfile, delimiter=',', quotechar='|')
			ii=0
			for row in reader:
				ii+=1
				if ii==1:
					continue
				#key=str(row[0].split('/')[-1]).split('_')[0]
				key=str(row[0].split('/')[1])
				value=float(row[1])
				if key in label_record:
					label_record[key].append(value)
				else:
					label_record[key]=list()
					label_record[key].append(value)

		for key in label_record.keys():
			self.label_dict[key]=np.median(label_record[key])

	def get_features(self):
		self.feature_dict=dict()
		for key in self.label_dict.keys():
			#key='2Yc6DqwE75w'
			#key='2Yc6DqwE75w_2_0_00_07.080000_0_00_10.405000'
			aggregated_feature_list=list()
			file_name='C:/Users/Yuchen/Downloads/french_features/French/French_OpenFace/'+key+".csv"
			if os.path.exists(file_name):
				#print(file_name)
				aggregated_feature_list=self.aggregate_features(file_name)
					
			if aggregated_feature_list:
				#print(aggregated_feature_list,len(aggregated_feature_list),len(aggregated_feature_list[0]))
				#print(len(np.mean(np.array(aggregated_feature_list),axis=0)))
				aggregated_feature=np.mean(np.array(aggregated_feature_list),axis=0)
			self.feature_dict[key]=aggregated_feature

	def aggregate_features(self,file_name):
		aggregated_feature_list=list()
		with open(file_name, newline='') as csvfile:
			reader = csv.reader(csvfile, delimiter=',', quotechar='|')
			ii=0
			for row in reader:
				ii+=1
				if ii==1:
					start_index=row.index(' AU01_r')
					if 'sucess' in row:
						success_index=row.index(' sucess')
					else:
						success_index=None
					continue
				else:
					if success_index==None or row[success_index]!=0:
						feature=[float(item) for item in row[start_index:start_index+17]] #[676:693]
						aggregated_feature_list.append(feature)
		return aggregated_feature_list

if __name__ == "__main__":
	a=build_moseas()
	#a.get_labels()
