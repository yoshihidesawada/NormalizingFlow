import sys
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset,DataLoader

from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal

import numpy as np


def gauss_sample(n_sample, dim):
	z = MultivariateNormal(torch.zeros(dim), torch.eye(dim))
	sampled_z = z.sample((n_sample,))

	plt.figure(figsize = (5,5))
	plt.xlim([-4, 4])
	plt.ylim([-4, 4])
	plt.scatter(sampled_z[:,0], sampled_z[:,1], s=15)
	plt.savefig('../outputs/gauss_repara.png')
	return z


def doublemoon_sample(n_sample):
	x1_1 = Normal(4, 4)
	sampled_x1_1 = x1_1.sample((int(n_sample/2),))
	x2_1 = Normal(0.25*(sampled_x1_1-4)**2-20, torch.ones_like(sampled_x1_1)*2)
	sampled_x2_1 = x2_1.sample()

	x1_2 = Normal(-4, 4)
	sampled_x1_2 = x1_2.sample((int(n_sample/2),))
	x2_2 = Normal(-0.25*(sampled_x1_2+4)**2+20, torch.ones_like(sampled_x1_2)*2)
	sampled_x2_2 = x2_2.sample()

	sampled_x1 = torch.cat([sampled_x1_1, sampled_x1_2])
	sampled_x2 = torch.cat([sampled_x2_1, sampled_x2_2])
	sampled_x = torch.zeros(n_sample, 2)
	sampled_x[:,0] = sampled_x1*0.2
	sampled_x[:,1] = sampled_x2*0.1

	labels = np.zeros((n_sample,2),dtype=np.float32)
	labels[:int(n_sample/2),0] = 1.0
	labels[int(n_sample/2):n_sample,1] = 1.0
	labels = torch.from_numpy(labels)

	plt.figure(figsize = (5,5))
	plt.xlim([-4, 4])
	plt.ylim([-4, 4])
	plt.scatter(sampled_x[:,0],sampled_x[:,1], s=15)
	plt.savefig('../outputs/doublemoon.png')

	return sampled_x, labels
