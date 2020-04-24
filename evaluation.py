import sys
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset,DataLoader

from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal

import numpy as np

import macro
import flow_model as flow
import dataset


def evaluation(model, model_type):

    if model_type == macro._JOINTREALNVP:
        prior_z = dataset.gauss_sample(n_sample=10000, dim=2+2)
    elif model_type == macro._CONDITIONALREALNVP:
        prior_z = dataset.gauss_sample(n_sample=10000, dim=2)

    sampled_x, sampled_labels = dataset.doublemoon_sample(n_sample=2000)
    test_loader = DataLoader(TensorDataset(sampled_x,sampled_labels), batch_size=64, shuffle=True)
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(test_loader):
            z, log_det_j_sum = model(data,labels)
            nll_loss = -(prior_z.log_prob(z)).mean()
            cur_loss = nll_loss.item()
            test_loss += cur_loss

        print('====> Average nll_loss: {:.4f}'.format(
            test_loss / (len(test_loader.dataset)*len(z[0]))
        ))



    prior_z = dataset.gauss_sample(n_sample=10000, dim=2)
    model.eval()
    with torch.no_grad():

        z = prior_z.sample((1000,))

        labels = np.zeros((1000,2),dtype=np.float32)
        labels[:,1] = 1.0
        labels = torch.from_numpy(labels)
        #print(labels.shape, z.shape)
        #labels = prior_z.sample((1000,))
        #print(labels.shape)

        x = model.backward(z,labels)
        z1 = z.numpy()
        x1 = x.numpy()


        z = prior_z.sample((1000,))
        labels = labels.numpy()
        labels[:,0] = 1.0
        labels[:,1] = 0.0
        labels = torch.from_numpy(labels)
        #labels = prior_z.sample((1000,))

        x = model.backward(z,labels)
        z0 = z.numpy()
        x0 = x.numpy()


        plt.figure(figsize = (5,5))
        plt.xlim([-4, 4])
        plt.ylim([-4, 4])
        plt.scatter(x0[:,0],x0[:,1], s=15, c='red')
        plt.scatter(x1[:,0],x1[:,1], s=15, c='blue')
        if model_type == macro._JOINTREALNVP:
            plt.savefig('./joint_generation_doublemoon.png')
        elif model_type == macro._CONDITIONALREALNVP:
            plt.savefig('./conditional_generation_doublemoon.png')

        plt.figure(figsize = (5,5))
        plt.xlim([-4, 4])
        plt.ylim([-4, 4])
        plt.scatter(z0[:,0],z0[:,1], s=15, c='red')
        plt.scatter(z1[:,0],z1[:,1], s=15, c='blue')
        plt.savefig('./gaussian.png')
