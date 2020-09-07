import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import pandas as pd
import logging
import os
import random
import math

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

def mask_operate(data, mask, col_ind):
	result = []
	for i in range(mask.shape[1]):
		sta = col_ind[i][0]
		end = col_ind[i][1]
		#data[:, sta:end] = data[:, sta:end]*mask[:, i:i+1]
		result.append(data[:, sta:end]*mask[:, i:i+1])
	result = torch.cat(result, dim = 1)
	return result

def compute_kl(real, pred):
	return torch.sum((torch.log(pred + 1e-4) - torch.log(real + 1e-4)) * pred)

def KL_Loss(x_fake, x_target, col_type, col_dim):
	kl = 0.0
	sta = 0
	end = 0
	for i in range(len(col_type)):
		dim = col_dim[i]
		sta = end
		end = sta+dim
		fakex = x_fake[:,sta:end]
		realx = x_target[:,sta:end]
		if col_type[i] == "gmm":
			fake2 = fakex[:,1:]
			real2 = realx[:,1:]
			dist = torch.sum(fake2, dim=0)
			dist = dist / torch.sum(dist)
			real = torch.sum(real2, dim=0)
			real = real / torch.sum(real)
			kl += compute_kl(real, dist)
		else:
			dist = torch.sum(fakex, dim=0)
			dist = dist / torch.sum(dist)
			
			real = torch.sum(realx, dim=0)
			real = real / torch.sum(real)
			
			kl += compute_kl(real, dist)
	return kl



class Handler:
	def __init__(self, mask_gen=None, mask_dis=None, obs_dis=None):
		self.mask_gen = mask_gen
		self.mask_dis = mask_dis
		self.obs_dis = obs_dis

	def train_joint(self, mask_gen, mask_dis, obs_dis, cp, dis_train_num, z_dim, epochs, steps_per_epoch, lr, sourceloader, targetloader, dataset, path, search, GPU=False):
		logging.basicConfig(filename=path+'joint_train_log_{}.log'.format(search), level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)
		itertimes = steps_per_epoch/50
		torch.manual_seed(0)
		torch.cuda.manual_seed(0)
		torch.cuda.manual_seed_all(0)
		np.random.seed(0)
		if GPU:
			mask_gen.GPU = True
			mask_gen.cuda()
			obs_dis.cuda()
			mask_dis.cuda()

		mask_D_optim = optim.RMSprop(mask_dis.parameters(), lr=lr, weight_decay=1e-5)
		obs_D_optim = optim.RMSprop(obs_dis.parameters(), lr=lr, weight_decay=1e-5)
		G_optim = optim.RMSprop(mask_gen.parameters(), lr=lr, weight_decay=1e-5)

		for epoch in range(epochs):
			logging.info("---------------------------------Epoch {}---------------------------------".format(epoch))
			print("---------------------------------Epoch {}---------------------------------".format(epoch))
			n_dis = 0
			for it in range(steps_per_epoch): 
				''' train Discriminator '''
				x_target, c_target, m_target = targetloader.sample(mask=True)
				x_source, c_source, m_source = sourceloader.sample(mask=True)
				x_target = mask_operate(x_target, m_target, dataset.col_ind)
				z = torch.randn(x_target.shape[0], z_dim)
				if GPU:
					z = z.cuda()
					x_target = x_target.cuda()
					x_source = x_source.cuda()
					m_target = m_target.cuda()

				m_fake = mask_gen(z, x_target)				
				ym_real = mask_dis(m_target, x_target)
				ym_fake = mask_dis(m_fake, x_target)	
				mask_D_Loss = -(torch.mean(ym_real) - torch.mean(ym_fake))
				G_optim.zero_grad()
				obs_D_optim.zero_grad()
				mask_D_optim.zero_grad()
				mask_D_Loss.backward()
				mask_D_optim.step()

				z = torch.randn(x_target.shape[0], z_dim)
				if GPU:
					z = z.cuda()
				m_fake = mask_gen(z, x_source)
				x_fake = mask_operate(x_source, m_fake, dataset.col_ind)
				yx_real = obs_dis(x_target)
				yx_fake = obs_dis(x_fake)
				obs_D_Loss = -(torch.mean(yx_real) - torch.mean(yx_fake))
				G_optim.zero_grad()
				obs_D_optim.zero_grad()
				mask_D_optim.zero_grad()
				obs_D_Loss.backward()
				obs_D_optim.step()

				for p in obs_dis.parameters():
					p.data.clamp_(-cp, cp)
				for p in mask_dis.parameters():
					p.data.clamp_(-cp, cp)
				n_dis += 1
				
				if n_dis == dis_train_num:
					n_dis = 0
					''' train Generator '''
					x_target, c_target, m_target = targetloader.sample(mask=True)
					x_source, c_source, m_source = sourceloader.sample(mask=True)
					x_target = mask_operate(x_target, m_target, dataset.col_ind)
					z1 = torch.randn(x_target.shape[0], z_dim)
					z2 = torch.randn(x_target.shape[0], z_dim)
					if GPU:
						z1 = z1.cuda()
						z2 = z2.cuda() 
						x_target = x_target.cuda()
						x_source = x_source.cuda()
						m_target = m_target.cuda()
					m_fake1 = mask_gen(z1, x_target)
					y_fake1 = mask_dis(m_fake1, x_target)
					G_Loss1 = -torch.mean(y_fake1)
					m_fake2 = mask_gen(z2, x_source)
					x_fake = mask_operate(x_source, m_fake2, dataset.col_ind)
					y_fake2 = obs_dis(x_fake)
					G_Loss2 = -torch.mean(y_fake2)
					G_Loss = G_Loss1 + G_Loss2
					G_optim.zero_grad()
					obs_D_optim.zero_grad()
					mask_D_optim.zero_grad()
					G_Loss.backward()
					G_optim.step()

				if it>=dis_train_num and it%itertimes == 0:
					logging.info("iterator {}, mask_D_Loss:{}, obs_D_Loss:{}, G_Loss:{}\n".format(it, mask_D_Loss.data, obs_D_Loss.data, G_Loss.data))
					print("iterator {}, mask_D_Loss:{}, obs_D_Loss:{}, G_Loss:{}\n".format(it, mask_D_Loss.data, obs_D_Loss.data, G_Loss.data))

			self.sample(mask_gen, z_dim, sourceloader, dataset, path+"sample_{}_{}".format(search, epoch), GPU=True, repeat=1)

		if GPU:
			mask_gen.GPU = False
			mask_gen.cpu()
			mask_dis.cpu()
			obs_dis.cpu()
		return mask_gen, mask_dis, obs_dis

	def train_mask(self, mask_gen, mask_dis, obs_dis, cp, dis_train_num, z_dim, epochs, steps_per_epoch, lr, sourceloader, targetloader, dataset, path, search, GPU=False):
		logging.basicConfig(filename=path+'mask_train_log_{}.log'.format(search), level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)
		itertimes = steps_per_epoch/50
		torch.manual_seed(0)
		torch.cuda.manual_seed(0)
		torch.cuda.manual_seed_all(0)
		np.random.seed(0)
		if GPU:
			mask_gen.GPU = True
			mask_gen.cuda()
			mask_dis.cuda()

		mask_D_optim = optim.RMSprop(mask_dis.parameters(), lr=lr, weight_decay=1e-5)
		G_optim = optim.RMSprop(mask_gen.parameters(), lr=lr, weight_decay=1e-5)

		for epoch in range(epochs):
			logging.info("---------------------------------Epoch {}---------------------------------".format(epoch))
			print("---------------------------------Epoch {}---------------------------------".format(epoch))
			n_dis = 0
			for it in range(steps_per_epoch): 
				''' train Discriminator '''
				x_target, c_target, m_target = targetloader.sample(mask=True)
				x_source, c_source, m_source = sourceloader.sample(mask=True)
				x_target = mask_operate(x_target, m_target, dataset.col_ind)
				z = torch.randn(x_target.shape[0], z_dim)
				if GPU:
					z = z.cuda()
					x_target = x_target.cuda()
					x_source = x_source.cuda()
					m_target = m_target.cuda()

				m_fake = mask_gen(z, x_target)				
				ym_real = mask_dis(m_target, x_target)
				ym_fake = mask_dis(m_fake, x_target)	
				mask_D_Loss = -(torch.mean(ym_real) - torch.mean(ym_fake))
				G_optim.zero_grad()
				mask_D_optim.zero_grad()
				mask_D_Loss.backward()
				mask_D_optim.step()

				for p in mask_dis.parameters():
					p.data.clamp_(-cp, cp)
				n_dis += 1
				
				if n_dis == dis_train_num:
					n_dis = 0
					''' train Generator '''
					x_target, c_target, m_target = targetloader.sample(mask=True)
					x_source, c_source, m_source = sourceloader.sample(mask=True)
					x_target = mask_operate(x_target, m_target, dataset.col_ind)
					z1 = torch.randn(x_target.shape[0], z_dim)
					if GPU:
						z1 = z1.cuda()
						x_target = x_target.cuda()
						x_source = x_source.cuda()
						m_target = m_target.cuda()
					m_fake1 = mask_gen(z1, x_target)
					y_fake1 = mask_dis(m_fake1, x_target)
					G_Loss1 = -torch.mean(y_fake1)
					G_Loss = G_Loss1
					G_optim.zero_grad()
					mask_D_optim.zero_grad()
					G_Loss.backward()
					G_optim.step()

				if it>=dis_train_num and it%itertimes == 0:
					logging.info("iterator {}, mask_D_Loss:{}, G_Loss:{}\n".format(it, mask_D_Loss.data, G_Loss.data))
					print("iterator {}, mask_D_Loss:{}, G_Loss:{}\n".format(it, mask_D_Loss.data, G_Loss.data))

			self.sample(mask_gen, z_dim, sourceloader, dataset, path+"sample_{}_{}".format(search, epoch), GPU=True, repeat=1)

		if GPU:
			mask_gen.GPU = False
			mask_gen.cpu()
			mask_dis.cpu()
		return mask_gen, mask_dis, obs_dis

	def train_obs(self, mask_gen, mask_dis, obs_dis, cp, dis_train_num, z_dim, epochs, steps_per_epoch, lr, sourceloader, targetloader, dataset, path, search, GPU=False):
		logging.basicConfig(filename=path+'obs_train_log_{}.log'.format(search), level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)
		itertimes = steps_per_epoch/50
		torch.manual_seed(0)
		torch.cuda.manual_seed(0)
		torch.cuda.manual_seed_all(0)
		np.random.seed(0)
		if GPU:
			mask_gen.GPU = True
			mask_gen.cuda()
			obs_dis.cuda()

		obs_D_optim = optim.RMSprop(obs_dis.parameters(), lr=lr, weight_decay=1e-5)
		G_optim = optim.RMSprop(mask_gen.parameters(), lr=lr, weight_decay=1e-5)

		for epoch in range(epochs):
			logging.info("---------------------------------Epoch {}---------------------------------".format(epoch))
			print("---------------------------------Epoch {}---------------------------------".format(epoch))
			n_dis = 0
			for it in range(steps_per_epoch): 
				''' train Discriminator '''
				x_target, c_target, m_target = targetloader.sample(mask=True)
				x_source, c_source, m_source = sourceloader.sample(mask=True)
				x_target = mask_operate(x_target, m_target, dataset.col_ind)
				z = torch.randn(x_target.shape[0], z_dim)
				if GPU:
					z = z.cuda()
					x_target = x_target.cuda()
					x_source = x_source.cuda()
					m_target = m_target.cuda()

				m_fake = mask_gen(z, x_source)
				x_fake = mask_operate(x_source, m_fake, dataset.col_ind)
				yx_real = obs_dis(x_target)
				yx_fake = obs_dis(x_fake)
				obs_D_Loss = -(torch.mean(yx_real) - torch.mean(yx_fake))
				G_optim.zero_grad()
				obs_D_optim.zero_grad()
				obs_D_Loss.backward()
				obs_D_optim.step()

				for p in obs_dis.parameters():
					p.data.clamp_(-cp, cp)
				n_dis += 1
				
				if n_dis == dis_train_num:
					n_dis = 0
					''' train Generator '''
					x_target, c_target, m_target = targetloader.sample(mask=True)
					x_source, c_source, m_source = sourceloader.sample(mask=True)
					x_target = mask_operate(x_target, m_target, dataset.col_ind)
					z2 = torch.randn(x_target.shape[0], z_dim)
					if GPU:
						z2 = z2.cuda() 
						x_target = x_target.cuda()
						x_source = x_source.cuda()
						m_target = m_target.cuda()
					m_fake2 = mask_gen(z2, x_source)
					x_fake = mask_operate(x_source, m_fake2, dataset.col_ind)
					y_fake2 = obs_dis(x_fake)
					G_Loss2 = -torch.mean(y_fake2)
					G_Loss = G_Loss2
					G_optim.zero_grad()
					obs_D_optim.zero_grad()
					G_Loss.backward()
					G_optim.step()

				if it>=dis_train_num and it%itertimes == 0:
					logging.info("iterator {}, obs_D_Loss:{}, G_Loss:{}\n".format(it, obs_D_Loss.data, G_Loss.data))
					print("iterator {}, obs_D_Loss:{}, G_Loss:{}\n".format(it, obs_D_Loss.data, G_Loss.data))

			self.sample(mask_gen, z_dim, sourceloader, dataset, path+"sample_{}_{}".format(search, epoch), GPU=True, repeat=1)

		if GPU:
			mask_gen.GPU = False
			mask_gen.cpu()
			obs_dis.cpu()
		return mask_gen, mask_dis, obs_dis

	def save(self, mname, mpath):
		torch.save(self.__dict__[mname], mpath)

	def load(self, mname, mpath):
		self.__dict__[mname] = torch.load(mpath)

	def sample(self, mask_gen, z_dim, sourceloader, dataset, path, GPU=True, repeat=1):
		mask_gen.eval()
		for time in range(repeat):
			sample_data = []
			for x_source, c_source, m_source in sourceloader:
				z = torch.randn(x_source.shape[0], z_dim)
				if GPU:
					z = z.cuda()
					x_source = x_source.cuda()
				m_fake = mask_gen(z, x_source)
				m_fake = torch.round(m_fake)
				m_fake = m_fake.cpu()
				m_fake = m_fake.detach().numpy()
				samples = x_source.cpu()
				sample_table = dataset.reverse(samples.detach().numpy())
				sample_table_none = dataset.reverse(samples.detach().numpy())
				m_fake = np.concatenate([m_fake, np.ones([m_fake.shape[0], sample_table.shape[1]-m_fake.shape[1]])], axis=1)
				for i in range(sample_table.shape[0]):
					for j in range(sample_table.shape[1]):
						if m_fake[i][j] == 0:
							sample_table[i][j] = 0.0
							sample_table_none[i][j] = None
				df = pd.DataFrame(sample_table, columns=dataset.columns)
				df_none = pd.DataFrame(sample_table_none, columns=dataset.columns)
				if len(sample_data) == 0:
					sample_data = df
					sample_data_none = df_none
				else:
					sample_data = sample_data.append(df)
					sample_data_none = sample_data_none.append(df_none)
			sample_data.to_csv(path+"_"+str(time)+".csv", index = None)
			sample_data_none.to_csv(path+"_"+str(time)+"_null.csv", index = None)
		mask_gen.train()
		
		
		
		
