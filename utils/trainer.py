import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
#import argparse
#import os
from torchvision import transforms
#from torch.jit import Error
#from torchvision.utils import save_image
from torchsummary import summary

"""
Find a data from here
https://archive.ics.uci.edu/ml/datasets.php
"""

class trainer(object):
    def __init__(self, model, train_loader, test_loader, batch_size = 16, log_interval = 10):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.eps = torch.finfo(float).eps #a small epsilon value
        self.model = model #input a model after initialize
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.batch_size = batch_size
        self.log_interval = log_interval

        self.criterion = nn.CrossEntropyLoss()
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def model_summary(self, forward_input = 2, dim = 10):
        if forward_input == 1:
            summary(self.model, input_size = (16,dim))
        elif forward_input == 2:
            summary(self.model, input_size = [(16,dim),(16,dim)])
        elif forward_input == 3:
            summary(self.model, input_size = [(16,dim),(16,dim),(16,dim)])

    def MIWAE_ELBO(self, outdic, indic = None):
        # the MIWAE ELBO 
        lpxz, lqzx, lpz = outdic['lpxz'], outdic['lqzx'], outdic['lpz'] 
        n_samples = torch.tensor([lpxz.shape[1]])
        l_w = lpxz + lpz - lqzx # importance weights in the paper eq(4) 
        log_sum_w = torch.logsumexp(l_w, dim=1) #dim=1: samples
        log_avg_weight = log_sum_w - torch.log(x = n_samples.type(torch.FloatTensor))
        #should be $l(\theda)$ in the paper
        # TODO: check self.n_samples should be one of output dimensions
        # .shape[]
        return torch.sum(log_avg_weight, -1)

    def gauss_loss(self, outdic, indic):
        x, m = indic['x'], indic['m']
        mu, log_sig = outdic['q_mu'], outdic['q_log_sig']
        #p(x | z) with Gauss z
        p_x_given_z = - (np.log(2 * np.pi) + log_sig + torch.square(x - mu) / (torch.exp(log_sig) + self.eps))/2.
        return torch.sum(p_x_given_z * m, -1)  # sum over d-dimension

    def bernoulli_loss(self, outdic, indic):
        x, m = indic['x'], indic['m']
        y = outdic['logits']
        #p(x | z) with bernoulli z
        p_x_given_z = x * torch.log(y + self.eps) + (1 - x) * torch.log(1 - y + self.eps)
        return torch.sum(m * p_x_given_z, -1)  # sum over d-dimension
        
    def KL_loss(self, outdic, indic = None):
        q_mu, q_log_sig = outdic['q_mu'], outdic['q_log_sig']
        KL = 1 + q_log_sig - torch.square(q_mu) - torch.exp(q_log_sig)
        return - torch.sum(KL, 1)/2.

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            outdic, q_z, out_sample = self.model(data)
            loss = self.MIWAE_ELBO(outdic)
            loss.backward()
            train_loss += loss
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.item() / len(data)))
                    
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(self.train_loader.dataset)))

    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(self.test_loader):
                data = data.to(self.device)
                outdic, q_z, out_sample = self.model(data)
                test_loss += self.MIWAE_ELBO(outdic)

        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))

