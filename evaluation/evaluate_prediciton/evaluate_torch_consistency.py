import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import sys
import argparse
import json
import multiprocessing
import os
import time
import logging
from pandas.api.types import is_object_dtype
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"


def feature_encoder(label_data):
	# transform categorical columns into numerical columns
	from sklearn.preprocessing import LabelEncoder
	from pandas.api.types import is_object_dtype
	label_con_data = label_data.copy()
	gens = {}
	for column in label_data.columns:
		if is_object_dtype(label_data[column]):
			gen_le = LabelEncoder()
			gen_labels = gen_le.fit_transform(list(label_data[column]))
			label_con_data.loc[:, column] = gen_labels  # to label from 0
			gens[column] = gen_le  # save the transformer to inverse
	# return a DataFrame
	return label_con_data, gens


def consistency_loss(y_adapt, y_train, y):
	loss1 = F.binary_cross_entropy(y_adapt, y) + F.binary_cross_entropy(y_train, y)
	loss2 = F.mse_loss(y_adapt, y_train) # 帮我confirm 下这里, OK
	return loss1 + loss2


def no_consistency_loss(y_adapt, y_train, y):
	loss1 = F.binary_cross_entropy(y_adapt, y) + F.binary_cross_entropy(y_train, y)
	# loss2 = F.mse_loss(y_adapt, y_train) # 帮我confirm 下这里, OK
	return loss1


class MLP(nn.Module):
	def __init__(self, shape):
		super(MLP, self).__init__()
		self.net = nn.Sequential(
			nn.Linear(shape, 50),
			nn.BatchNorm1d(50),
			nn.ReLU(),

			nn.Linear(50, 25),
			nn.BatchNorm1d(25),
			nn.ReLU(),

			nn.Linear(25, 1),
			nn.Sigmoid()
		)

	def forward(self, x):
		return self.net(x)


def train(model, adapt_x, train_x, train_y, l2, epochs, lr, consistency):
	model.train()
	# logging.basicConfig(filename=prefix+'_log.log', level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)
	optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
	train_y = train_y.view(-1, 1)  # 这是为啥
	for epoch in range(epochs):
		y_pred_train = model(train_x)
		y_pred_adapt = model(adapt_x)
		if consistency == 'yes':
			loss = consistency_loss(y_pred_adapt, y_pred_train, train_y)
		else:
			loss = no_consistency_loss(y_pred_adapt, y_pred_train, train_y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if epoch % 100 == 0:
			logging.info("iterator {}, Loss:{}".format(epoch, loss.data))

	model.eval()
	return model


def evaluation(y_true, y_pred, y_prob, threshold=None, parameters=None):
	# y_true is list of true labels
	# y_pred is list of predicted labels
	# y_prob is list of probabilities
	# threshold is the number by which we transform probability to one label
	# parameters is the parameters of the model
	from sklearn import metrics
	f1_score = metrics.f1_score(y_true, y_pred)  # exert the same importance on precision and recall
	precision = metrics.precision_score(y_true, y_pred)
	recall = metrics.recall_score(y_true, y_pred)
	balanced_f1_score = metrics.f1_score(y_true, y_pred, average='weighted')
	balanced_precision = metrics.precision_score(y_true, y_pred, average='weighted')
	balanced_recall = metrics.recall_score(y_true, y_pred, average='weighted')
	#return [recall, precision, f1_score, balanced_recall, balanced_precision, balanced_f1_score, threshold, parameters]
	return [f1_score, balanced_f1_score]


def result_sum(mlp_val, name):
	# name is the name of output file
	val_result = pd.DataFrame({'Neural Network': mlp_val}, index=['f1_score', 'balanced f1_score'])
	# line = "{},{},{}\n".format(mlp_val[0], mlp_val[1], mlp_val[2])
	# with open(name, "a") as f:
	# 	f.write(line)
	val_result.to_csv(name, encoding='utf_8_sig')
	print('the results are in %s' % name)


def thread_run(config, adapt_data, train_data, test_data):
	# test_data = test_data.fillna(0.0)
	# print(train_file)
	# train_data = train_data.fillna(0.0)
	data = pd.concat([adapt_data, train_data, test_data], keys=['adapt', 'train', 'test'])
	# drop unnecessary columns
	try:
		data = data.drop(columns=['Unnamed: 0'])
	except:
		pass
	try:
		data = data.drop(columns=['Unnamed: 0.1'])
	except:
		pass
	try:
		data = data.drop(columns=['Unnamed: 0.1.1'])
	except:
		pass
	# transform the categorical columns into numerical columns
	featured_con_data, gens = feature_encoder(data)

	# split train and test from data
	label_column = config["label_column"]

	adapt_data = featured_con_data.loc['adapt']
	train_data = featured_con_data.loc['train']
	test_data = featured_con_data.loc['test']
	
	X_adapt = adapt_data.drop(axis=1, columns=[label_column])
	X_train = train_data.drop(axis=1, columns=[label_column])
	train_y = train_data[label_column]

	X_test = test_data.drop(axis=1, columns=[label_column])
	test_y = test_data[label_column]
	
	# scaling
	from sklearn import preprocessing
	scaled_test_x = preprocessing.scale(X_test)
	print('scaled test x shape:', scaled_test_x.shape[1])
	scaled_train_x = preprocessing.scale(X_train)
	print('scaled train x shape:', scaled_train_x.shape[1])
	scaled_adapt_x = preprocessing.scale(X_adapt)
	print('scaled adapt x shape:', scaled_adapt_x.shape[1])

	# neural network
	# create the model
	model = MLP(config["input_shape"])

	scaled_adapt_x = torch.from_numpy(scaled_adapt_x).float().cuda()
	scaled_train_x = torch.from_numpy(scaled_train_x).float().cuda()
	train_y = torch.from_numpy(train_y.values).float().cuda()
	scaled_test_x = torch.from_numpy(scaled_test_x).float()
	test_y = torch.from_numpy(test_y.values).float()
	
	# for l2 in config["l2"]:
	model.cuda()
	model = train(model, scaled_adapt_x, scaled_train_x, train_y, config["l2"], config["epoch_space"], 0.01, config["consistency"])
	model.eval()
	model.cpu()
	mlp_prob_test_y = model(scaled_test_x)
	mlp_pred_test_y = (mlp_prob_test_y > 0.5) + 0
	mlp_pred_test_y = mlp_pred_test_y.cpu()

	print('Neural network model training ends.')
	# evaluate
	mlp_val = evaluation(test_y, mlp_pred_test_y, mlp_prob_test_y)  #mlp_thres, mlp_para)
	# mlp_val = ["{}_{}".format(train_file,config["l2"])]+mlp_val
	result_sum(mlp_val, config["name"])


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('configs', help='a json config file')
	parser.add_argument('gpu', default=0)
	args = parser.parse_args()
	gpu = int(args.gpu)
	if gpu >= 0:
		GPU = True
		os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
	else:
		GPU = False

	with open(args.configs) as f:
		configs = json.load(f)

	jobs = []
	for config in configs:
		train_data = pd.read_csv(config["train_data"])
		test_data = pd.read_csv(config["test_data"])

		adapt_data = pd.read_csv(config["adapt_data"])

		j = multiprocessing.Process(target=thread_run, args=(config, adapt_data, train_data, test_data))
		jobs.append(j)
		j.start()
		if len(jobs) == 20:
			for j in jobs:
				j.join()
			jobs = []
