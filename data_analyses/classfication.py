import torch
import os
import sys
import numpy as np
from dataloader import build_mit_swav, build_moseas
from model.MLP import MLP

def partition_dataset(n, proportion=0.8):
	train_num = int(n * proportion)
	indices = np.random.permutation(n)
	train_indices, val_indices = indices[:train_num], indices[train_num:]
	return train_indices, val_indices

if __name__ == "__main__":

	# file_dir='C:/Users/Yuchen/Documents/verb-alignment_new/data/dumped_embeddings/mit_swav_100.pkl'
	# train_dataset=build_mit_swav(file_dir)
	# test_dataset=build_mit_swav(file_dir)

	print('loading data...')
	train_dataset=build_moseas()
	test_dataset=build_moseas()

	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

	# dataset=build_moseas()

	# train_indices = torch.randperm(len(dataset))[:10]
	# test_indices = torch.randperm(len(dataset))[:10]

	# dataset_1 = torch.utils.data.Subset(dataset, train_indices)
	# dataset_2 = torch.utils.data.Subset(dataset, test_indices)
	

	# train_loader = torch.utils.data.DataLoader(dataset_1, batch_size=5, shuffle=True)
	# test_loader = torch.utils.data.DataLoader(dataset_2, batch_size=5, shuffle=False)
	print('finish.')


	model=MLP(input_size=17,hidden_size=50,output_size=7,epochs=10000)

	print("TRAINING.")
	model.train(train_loader, test_loader)

	print("TEST.")
	test_losses=model.test(test_loader)
	print("test loss:",np.mean(test_losses))