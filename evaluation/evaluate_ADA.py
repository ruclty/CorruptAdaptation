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
from evaluate_preprocess import data_processing
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
	loss1 = F.binary_cross_entropy(y_train, y)
	loss2 = F.binary_cross_entropy(y_adapt, y) 
	loss3 = F.mse_loss(y_adapt, y_train) # 帮我confirm 下这里, OK
	return loss1, loss2, loss3


def no_consistency_loss(y_adapt, y_train, y):
	loss1 = F.binary_cross_entropy(y_train, y)
	loss2 = F.binary_cross_entropy(y_adapt, y)
	return loss1, loss2


class MLP(nn.Module):
	def __init__(self, shape, multiclass=False):
		super(MLP, self).__init__()
		self.net = nn.Sequential(
			nn.Linear(shape, 100),
			nn.BatchNorm1d(100),
			nn.ReLU(),

			nn.Linear(100, 50),
			nn.BatchNorm1d(50),
			nn.ReLU(),

			nn.Linear(50, 1),
			nn.Sigmoid()
		)

	def forward(self, x):
		return self.net(x)

def train_simple(model, train_x, train_y, test_x, test_y, valid_x, valid_y, l2, epochs, lr):
	model.train()
	optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
	mlp_vals = []
	num = epochs/10
	for epoch in range(epochs):
		y_ = model(train_x)
		loss = F.binary_cross_entropy(y_, train_y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if epoch % 100 == 0 and epoch != 0:
			model.eval()
			print("iterator {}, Loss:{}".format(epoch, loss.data))
			logging.info("iterator {}, Loss:{}".format(epoch, loss.data))

			mlp_prob_valid_y = model(valid_x)
			mlp_pred_valid_y = (mlp_prob_valid_y > 0.5) + 0
			mlp_pred_valid_y = mlp_pred_valid_y.cpu()
			mlp_val_valid = evaluation(valid_y.cpu(), mlp_pred_valid_y.cpu(), mlp_prob_valid_y.cpu())

			mlp_prob_test_y = model(test_x)
			mlp_pred_test_y = (mlp_prob_test_y > 0.5) + 0
			mlp_pred_test_y = mlp_pred_test_y.cpu()
			mlp_val_test = evaluation(test_y.cpu(), mlp_pred_test_y.cpu(), mlp_prob_test_y.cpu())  #mlp_thres, mlp_para)
			
			mlp_vals.append(["l2={},epoch={}".format(l2, epoch)]+mlp_val_valid+mlp_val_test)
			model.train()
	model.eval()
	return model, mlp_vals


def train_with_adapt(model, adapt_x, train_x, train_y, l2, epochs, lr, consistency):
	model.train()
	optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)  
	for epoch in range(epochs):
		y_pred_train = model(train_x)
		y_pred_adapt = model(adapt_x)
		if consistency:
			loss1, loss2, loss3 = consistency_loss(y_pred_adapt, y_pred_train, train_y)
			loss = loss1 + loss2 + loss3
		else:
			loss1, loss2 = no_consistency_loss(y_pred_adapt, y_pred_train, train_y)
			loss = loss1 + loss2
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if epoch % 300 == 0:
			if consistency:
				print("iterator {}, train_Loss:{}, adapt_Loss:{}, consistency_loss:{}".format(epoch, loss1.data, loss2.data, loss3.data))
				logging.info("iterator {}, train_Loss:{}, adapt_Loss:{}, consistency_loss:{}".format(epoch, loss1.data, loss2.data, loss3.data))
			else:
				print("iterator {}, train_Loss:{}, adapt_Loss:{}".format(epoch, loss1.data, loss2.data))
				logging.info("iterator {}, train_Loss:{}, adapt_Loss:{}".format(epoch, loss1.data, loss2.data))
	model.eval()
	return model


def evaluation(y_true, y_pred, y_prob, threshold=None, parameters=None):
	from sklearn import metrics
	f1_score = metrics.f1_score(y_true, y_pred)  
	precision = metrics.precision_score(y_true, y_pred)
	recall = metrics.recall_score(y_true, y_pred)
	balanced_f1_score = metrics.f1_score(y_true, y_pred, average='weighted')
	balanced_precision = metrics.precision_score(y_true, y_pred, average='weighted')
	balanced_recall = metrics.recall_score(y_true, y_pred, average='weighted')
	return [f1_score]


def train_classifiers_simple(config, train_data, test_data):
	data = pd.concat([train_data, test_data], keys=['train', 'test'])
	featured_con_data, gens = feature_encoder(data)

	label_column = config["label_column"]
	train_data = featured_con_data.loc['train']
	test_data = featured_con_data.loc['test']
	X_train = train_data.drop(axis=1, columns=[label_column])
	train_Y = train_data[label_column]
	X_test = test_data.drop(axis=1, columns=[label_column])
	test_Y = test_data[label_column]

	from sklearn import preprocessing
	length = int(len(X_train)/4)

	scaled_test_x = preprocessing.scale(X_test)
	scaled_train_x = preprocessing.scale(X_train.iloc[:length*3,:])
	scaled_valid_x = preprocessing.scale(X_train.iloc[length*3:,:])
	train_y = train_Y.iloc[:length*3]
	valid_y = train_Y.iloc[length*3:]

	print(scaled_valid_x.shape, scaled_train_x.shape, scaled_test_x.shape)

	scaled_valid_x = torch.from_numpy(scaled_valid_x).float().cuda()
	scaled_train_x = torch.from_numpy(scaled_train_x).float().cuda()
	train_y = torch.from_numpy(train_y.values).float().cuda()
	valid_y = torch.from_numpy(valid_y.values).float().cuda()
	scaled_test_x = torch.from_numpy(scaled_test_x).float().cuda()
	test_y = torch.from_numpy(test_Y.values).float().cuda()

	mlp_vals = []
	for l2 in config["l2"]:	
		model = MLP(config["input_shape"])
		model.cuda()
		model, mlp_val = train_simple(model, scaled_train_x, train_y, scaled_test_x, test_y, scaled_valid_x, valid_y, l2, config["epoch"], 0.01)
		if len(mlp_vals) == 0:
			mlp_vals = mlp_val
		else:
			mlp_vals = mlp_vals + mlp_val
	return mlp_vals


def train_classifiers_with_adapt(config, adapt_data, train_data, test_data, consistency):
	data = pd.concat([adapt_data, train_data, test_data], keys=['adapt', 'train', 'test'])

	featured_con_data, gens = feature_encoder(data)
	label_column = config["label_column"]

	adapt_data = featured_con_data.loc['adapt']
	train_data = featured_con_data.loc['train']
	test_data = featured_con_data.loc['test']
	
	X_adapt = adapt_data.drop(axis=1, columns=[label_column])
	X_train = train_data.drop(axis=1, columns=[label_column])
	train_y = train_data[label_column]

	X_test = test_data.drop(axis=1, columns=[label_column])
	test_y = test_data[label_column]

	from sklearn import preprocessing
	scaled_test_x = preprocessing.scale(X_test)
	scaled_train_x = preprocessing.scale(X_train)
	scaled_adapt_x = preprocessing.scale(X_adapt)
	model = MLP(config["input_shape"])

	scaled_adapt_x = torch.from_numpy(scaled_adapt_x).float().cuda()
	scaled_train_x = torch.from_numpy(scaled_train_x).float().cuda()
	train_y = torch.from_numpy(train_y.values).float().cuda()
	scaled_test_x = torch.from_numpy(scaled_test_x).float()
	test_y = torch.from_numpy(test_y.values).float()

	model.cuda()
	model = train_with_adapt(model, scaled_adapt_x, scaled_train_x, train_y, config["l2"], config["epoch_space"], 0.001, consistency)
	model.eval()
	model.cpu()
	mlp_prob_test_y = model(scaled_test_x)
	mlp_pred_test_y = (mlp_prob_test_y > 0.5) + 0
	mlp_pred_test_y = mlp_pred_test_y.cpu()
	mlp_val = evaluation(test_y, mlp_pred_test_y, mlp_prob_test_y)  #mlp_thres, mlp_para)
	return mlp_val

def thread_run(config):
	logging.basicConfig(filename=config["output"]+'_log.log', level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)
	noise_list = config["noise_list"]

	test_dir = config["test_dir"]
	train_dir = config["train_dir"]
	adapt_dir = config["adapt_dir"]
	gadapt_dir = config["gadapt_dir"]
	gan_dir = config["gan_dir"]

	logging.info(config["train_method"])	
	for noise in noise_list:
		test_data = pd.read_csv(test_dir+"test_{}.csv".format(noise))
		logging.info(test_dir+"test_{}.csv".format(noise))
		if config["run"] == "train":
			path = train_dir+"train_0.6_{}.csv".format(noise)
		elif config["run"] == "kadapt":
			path = adapt_dir+"kadapt_{}.csv".format(noise)
		elif config["run"] == "clean":
			path = config["clean_path"]
		elif config["run"] == "gan":
			path = gadapt_dir+"gadapt_{}.csv".format(noise)
		elif config["run"] == "gan_nomask":
			path = gan_dir+"impute_gadapt_{}.csv".format(noise)
		logging.info(path)
		train_data = pd.read_csv(path)
		mlp_vals = train_classifiers_simple(config, train_data, test_data)
		for i, mlp_val in enumerate(mlp_vals):
			mlp_vals[i] = [path]+mlp_val
		df = pd.DataFrame(mlp_vals, columns=["file", "l2", "f1_valid", "f1_test"])
		df.to_csv(config["output"]+"_{}.csv".format(noise), index=None)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('configs', help='a json config file')
	parser.add_argument('gpu', default=0)
	args = parser.parse_args()
	gpu = int(args.gpu)
	torch.manual_seed(0)
	torch.cuda.manual_seed(0)
	torch.cuda.manual_seed_all(0)
	np.random.seed(0)
	if gpu >= 0:
		GPU = True
		os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
	else:
		GPU = False

	with open(args.configs) as f:
		configs = json.load(f)

	jobs = []
	for config in configs:
		thread_run(config)