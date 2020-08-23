from data import NumericalField, CategoricalField, Iterator
from data import Dataset
from synthesizer import MaskGenerator_MLP, ObservedGenerator_MLP, Discriminator, Handler, ObservedGenerator_LSTM
from random import choice
import multiprocessing
import pandas as pd
import numpy as np
import torch
import argparse
import json
import os

parameters_space = {
	"batch_size":[256, 512],
	"z_dim":[100, 200, 300], 
	"gen_num_layers":[1,2,3],
	"gen_hidden_dim":[100, 200, 300, 400],
	"gen_feature_dim":[100,200],
	"gen_lstm_dim":[100,200,300,400,500],
	"dis_hidden_dim":[100, 200, 300],
	"dis_num_layers":[1,2,3],
	"lr":[0.0001],
	"cp":[0.01],
	"dis_train_num" :[1, 2, 5]
}

def parameter_search(gen_model):
	param = {}
	param["batch_size"] = choice(parameters_space["batch_size"])
	param["z_dim"] = choice(parameters_space["z_dim"])

	if gen_model == "MLP":
		param["gen_hidden_dims"] = []
		gen_num_layers = choice(parameters_space["gen_num_layers"])
		for l in range(gen_num_layers):
			dim = choice(parameters_space["gen_hidden_dim"])
			if l > 0 and param["gen_hidden_dims"][l-1] > dim:
				dim = param["gen_hidden_dims"][l-1]
			param["gen_hidden_dims"].append(dim)

	elif gen_model == "LSTM":
		param["gen_feature_dim"] = choice(parameters_space["gen_feature_dim"])
		param["gen_lstm_dim"] = choice(parameters_space["gen_lstm_dim"])

	param["obs_dis_hidden_dims"] = []
	dis_num_layers = choice(parameters_space["dis_num_layers"])
	for l in range(dis_num_layers):
		dim = choice(parameters_space["dis_hidden_dim"])
		if l > 0 and param["obs_dis_hidden_dims"][l-1] < dim:
			dim = param["obs_dis_hidden_dims"][l-1]
		param["obs_dis_hidden_dims"].append(dim)

	param["mask_dis_hidden_dims"] = []
	dis_num_layers = choice(parameters_space["dis_num_layers"])
	for l in range(dis_num_layers):
		dim = choice(parameters_space["dis_hidden_dim"])
		if l > 0 and param["mask_dis_hidden_dims"][l-1] < dim:
			dim = param["mask_dis_hidden_dims"][l-1]
		param["mask_dis_hidden_dims"].append(dim)

	param["lr"] = choice(parameters_space["lr"])
	param["cp"] = choice(parameters_space["cp"])
	param["dis_train_num"] = choice(parameters_space["dis_train_num"])
	return param

def thread_run(path, search, config, source_dst, target_dst, GPU):
	if config["rand_search"] == "yes":
		param = parameter_search(gen_model=config["gen_model"])
	else:
		param = config["param"]
		
	with open(path+"exp_params.json", "a") as f:
		json.dump(param, f)
		f.write("\n")
		

	source_it = Iterator(dataset=source_dst, batch_size=20, shuffle=True)
	target_it = Iterator(dataset=target_dst, batch_size=20, shuffle=True, mask=config["target_mask"])

	x_dim = source_it.data.shape[1]
	col_ind = source_dst.col_ind
	col_dim = source_dst.col_dim
	col_type = source_dst.col_type
	mask_dim = target_it.masks.shape[1]

	if config["gen_model"] == "LSTM":
		mask_gen = ObservedGenerator_LSTM(param["zx_dim"], param["obs_gen_feature_dim"], param["obs_gen_lstm_dim"], col_dim, col_type, c_dim)
	elif config["gen_model"] == "MLP":
		mask_gen = MaskGenerator_MLP(param["z_dim"], x_dim, param["gen_hidden_dims"], mask_dim)

	obs_dis = Discriminator(x_dim, param["obs_dis_hidden_dims"])
	mask_dis = Discriminator(mask_dim, param["mask_dis_hidden_dims"], c_dim=x_dim, condition=True)

	print(mask_gen)
	print(mask_dis)
	print(obs_dis)

	handler = Handler(mask_gen=mask_gen, mask_dis=mask_dis, obs_dis=obs_dis)

	if "training" not in config.keys():
		config["training"] = "Joint"
	if config["training"] == "Joint":
		mask_gen, mask_dis, obs_dis = handler.train_joint(mask_gen, mask_dis, obs_dis, param["cp"], param["dis_train_num"], param["z_dim"], config["epochs"], config["steps_per_epoch"], param["lr"], source_it, target_it, source_dst, path, search, GPU=GPU)
	elif config["training"] == "Mask":
		mask_gen, mask_dis, obs_dis = handler.train_mask(mask_gen, mask_dis, obs_dis, param["cp"], param["dis_train_num"], param["z_dim"], config["epochs"], config["steps_per_epoch"], param["lr"], source_it, target_it, source_dst, path, search, GPU=GPU)	
	elif config["training"] == "Observer":
		mask_gen, mask_dis, obs_dis = handler.train_obs(mask_gen, mask_dis, obs_dis, param["cp"], param["dis_train_num"], param["z_dim"], config["epochs"], config["steps_per_epoch"], param["lr"], source_it, target_it, source_dst, path, search, GPU=GPU)	

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
	try:
		os.mkdir("expdir")
	except:
		pass
		
	for config in configs:
		path = "expdir/"+config["name"]+"/"
		try:
			os.mkdir("expdir/"+config["name"])
		except:
			pass
		source = pd.read_csv(config["source"])
		target = pd.read_csv(config["target"])
		
		fields = []
		col_type = []
		if "label" in config.keys():
			cond = config["label"]
		for i, col in enumerate(list(source)):
			if "label" in config.keys() and col in cond:
				fields.append((col, CategoricalField("one-hot", noise=0)))
				col_type.append("condition")
			elif i in config["normalize_cols"]:
				fields.append((col,NumericalField("normalize")))
				col_type.append("normalize")
			elif i in config["gmm_cols"]:
				fields.append((col, NumericalField("gmm", n=5)))
				col_type.append("gmm")
			elif i in config["one-hot_cols"]:
				fields.append((col, CategoricalField("one-hot", noise=0)))
				col_type.append("one-hot")
			elif i in config["ordinal_cols"]:
				fields.append((col, CategoricalField("dict")))
				col_type.append("ordinal")

		source_dst, target_dst = Dataset.split(
			fields = fields,
			path = ".",
			col_type = col_type,
			train = config["source"],
			validation = config["target"],
			format = "csv",
		)
		source_dst.learn_convert()
		target_dst.learn_convert()

		print("source row : {}".format(len(source_dst)))
		print("target row: {}".format(len(target_dst)))
		
		n_search = config["n_search"]
		jobs = [multiprocessing.Process(target=thread_run, args=(path, search, config, source_dst, target_dst, GPU)) for search in range(n_search)]	
		for j in jobs:
			j.start()
		for j in jobs:
			j.join()
				









