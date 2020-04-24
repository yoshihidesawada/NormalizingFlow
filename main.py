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
import dataset
import flow_model as flow
import evaluation as eval

if len(sys.argv) != 2:
    print("error")
    sys.exit(1)
_, model_type = sys.argv


def main(model_type):

    if model_type == macro._JOINTREALNVP:
        prior_z = dataset.gauss_sample(n_sample=10000, dim=2+2)
        mask = torch.from_numpy(np.array([0, 1, 0, 1]).astype(np.float32))
        model = flow.JointRealNVP(input_dim=2+2, output_dim=2+2, hid_dim=512, mask=mask, n_layers=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    elif model_type == macro._CONDITIONALREALNVP:
        prior_z = dataset.gauss_sample(n_sample=10000, dim=2)
        mask = torch.from_numpy(np.array([0, 1]).astype(np.float32))
        model = flow.ConditionalRealNVP(input_dim=2+2, output_dim=2, hid_dim=512, mask=mask, n_layers=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    sampled_x, sampled_labels = dataset.doublemoon_sample(n_sample=10000)
    train_loader = DataLoader(TensorDataset(sampled_x,sampled_labels), batch_size=64, shuffle=True)

    model.train()
    train_loss = 0
    for epoch in range(macro._EPOCH):
        for batch_idx, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            z, log_det_j_sum = model(data,labels)

            if model_type == macro._JOINTREALNVP:
                y = z[:,len(z[0])-len(labels[0]):]
                log_prob_loss = -(prior_z.log_prob(z)+log_det_j_sum).mean()
                mse_loss = F.mse_loss(y, labels)
                loss = log_prob_loss + macro._LAMBDA*mse_loss
            elif model_type == macro._CONDITIONALREALNVP:
                loss = -(prior_z.log_prob(z)+log_det_j_sum).mean()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print('Epoch: {} Average loss: {:.4f}'.format(\
        epoch, train_loss / (len(train_loader.dataset)*len(z[0]))))

    eval.evaluation(model, model_type)







if __name__ == '__main__':
    model_type = int(model_type)
    main(model_type)
