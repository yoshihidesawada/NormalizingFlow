import sys

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset,DataLoader

from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal

import numpy as np

class ConditionalRealNVP(nn.Module):
	def __init__(self, input_dim, output_dim, hid_dim, mask, n_layers):
		super().__init__()
		self.modules = []
		self.modules.append(ConditionalCouplingLayer(input_dim, output_dim, hid_dim, mask))
		for _ in range(n_layers-2):
			mask = 1 - mask
			self.modules.append(ConditionalCouplingLayer(input_dim, output_dim, hid_dim, mask))
		self.modules.append(ConditionalCouplingLayer(input_dim, output_dim, hid_dim, 1 - mask))
		self.module_list = nn.ModuleList(self.modules)

	def forward(self, x, labels):
		ldj_sum = 0 # sum of log determinant of jacobian
		for module in self.module_list:
			x, ldj= module(x, labels)
			ldj_sum += ldj
		return x, ldj_sum

	def backward(self, z, labels):
		for module in reversed(self.module_list):
			z = module.backward(z,labels)
		return z

class JointRealNVP(nn.Module):
	def __init__(self, input_dim, output_dim, hid_dim, mask, n_layers):
		super().__init__()
		self.modules = []
		self.modules.append(JointCouplingLayer(input_dim, output_dim, hid_dim, mask))
		for _ in range(n_layers-2):
			mask = 1 - mask
			self.modules.append(JointCouplingLayer(input_dim, output_dim, hid_dim, mask))
		self.modules.append(JointCouplingLayer(input_dim, output_dim, hid_dim, 1 - mask))
		self.module_list = nn.ModuleList(self.modules)

	def forward(self, x, labels):
		ldj_sum = 0 # sum of log determinant of jacobian
		x_cat = torch.cat([x, labels],dim=1)
		for module in self.module_list:
			x_cat, ldj= module(x_cat)
			ldj_sum += ldj
		return x_cat, ldj_sum

	def backward(self, z, labels):
		z_cat = torch.cat([z, labels],dim=1)
		print(z_cat.shape, z.shape, labels.shape)
		for module in reversed(self.module_list):
			z_cat = module.backward(z_cat)
		return z_cat


class ConditionalCouplingLayer(nn.Module):
	def __init__(self, input_dim, output_dim, hid_dim, mask):
		super().__init__()
		self.s_fc1 = nn.Linear(input_dim, hid_dim)
		self.s_fc2 = nn.Linear(hid_dim, hid_dim)
		self.s_fc3 = nn.Linear(hid_dim, output_dim)
		self.t_fc1 = nn.Linear(input_dim, hid_dim)
		self.t_fc2 = nn.Linear(hid_dim, hid_dim)
		self.t_fc3 = nn.Linear(hid_dim, output_dim)
		self.mask = mask

	def forward(self, x, labels):
		x_mask = x * self.mask
		x_m = torch.cat([x_mask, labels], dim=1)
		s_out = torch.tanh(self.s_fc3(F.relu(self.s_fc2(F.relu(self.s_fc1(x_m))))))
		t_out = self.t_fc3(F.relu(self.t_fc2(F.relu(self.t_fc1(x_m)))))
		y = x_mask + (1-self.mask)*(x*torch.exp(s_out)+t_out)
		log_det_jacobian = s_out.sum(dim=1)
		return y, log_det_jacobian

	def backward(self, y, labels):
		y_mask = y * self.mask
		y_m = torch.cat([y_mask, labels], dim=1)
		s_out = torch.tanh(self.s_fc3(F.relu(self.s_fc2(F.relu(self.s_fc1(y_m))))))
		t_out = self.t_fc3(F.relu(self.t_fc2(F.relu(self.t_fc1(y_m)))))
		x = y_mask + (1-self.mask)*(y-t_out)*torch.exp(-s_out)
		return x

class JointCouplingLayer(nn.Module):
	def __init__(self, input_dim, output_dim, hid_dim, mask):
		super().__init__()
		self.s_fc1 = nn.Linear(input_dim, hid_dim)
		self.s_fc2 = nn.Linear(hid_dim, hid_dim)
		self.s_fc3 = nn.Linear(hid_dim, output_dim)
		self.t_fc1 = nn.Linear(input_dim, hid_dim)
		self.t_fc2 = nn.Linear(hid_dim, hid_dim)
		self.t_fc3 = nn.Linear(hid_dim, output_dim)
		self.mask = mask

	def forward(self, x):
		x_m = x * self.mask
		s_out = torch.tanh(self.s_fc3(F.relu(self.s_fc2(F.relu(self.s_fc1(x_m))))))
		t_out = self.t_fc3(F.relu(self.t_fc2(F.relu(self.t_fc1(x_m)))))
		y = x_m + (1-self.mask)*(x*torch.exp(s_out)+t_out)
		log_det_jacobian = s_out.sum(dim=1)
		return y, log_det_jacobian

	def backward(self, y):
		y_m = y * self.mask
		s_out = torch.tanh(self.s_fc3(F.relu(self.s_fc2(F.relu(self.s_fc1(y_m))))))
		t_out = self.t_fc3(F.relu(self.t_fc2(F.relu(self.t_fc1(y_m)))))
		x = y_m + (1-self.mask)*(y-t_out)*torch.exp(-s_out)
		return x
