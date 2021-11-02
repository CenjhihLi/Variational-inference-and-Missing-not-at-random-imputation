import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as pdfun
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

class VAEtrainer(object):
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

    def model_summary(self, forward_input = 1, dim = 10):
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
        x = indic['x']
        m = np.array(np.isnan(x), dtype=np.float32)
        mu, log_sig = outdic['q_mu'], outdic['q_log_sig']
        #p(x | z) with Gauss z
        p_x_given_z = - (np.log(2 * np.pi) + log_sig + torch.square(x - mu) / (torch.exp(log_sig) + self.eps))/2.
        return torch.sum(p_x_given_z * m, -1)  # sum over d-dimension

    def bernoulli_loss(self, outdic, indic):
        x = indic['x']
        m = np.array(np.isnan(x), dtype=np.float32)
        y = outdic['logits']
        #p(x | z) with bernoulli z
        p_x_given_z = x * torch.log(y + self.eps) + (1 - x) * torch.log(1 - y + self.eps)
        return torch.sum(m * p_x_given_z, -1)  # sum over d-dimension
        
    def KL_loss(self, outdic, indic = None):
        q_mu, q_log_sig = outdic['q_mu'], outdic['q_log_sig']
        KL = 1 + q_log_sig - torch.square(q_mu) - torch.exp(q_log_sig)
        return - torch.sum(KL, 1)/2.
    
    def VAE_loss(self, outdic, indic):
        recon_x, x = indic['recon_x'], indic['x']
        q_mu, q_log_sig = outdic['q_mu'], outdic['q_log_sig']
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        # KLD = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + q_log_sig - torch.square(q_mu) - torch.exp(q_log_sig))
        return BCE + KLD
    
    def imputation(self, m, x):
        """
        x: Xz[np.isnan(Xnan)] = 0
        """
        outdic, q_z, l_out_sample  = self.model(x)
        lpxz, lqzx, lpz = outdic['lpxz'], outdic['lqzx'], outdic['lpz'] 
        l_w = lpxz + lpz - lqzx # importance weights in the paper eq(4) 
        wl = F.softmax(l_w, dim = 1) #TODO: check
        xm = np.sum((l_out_sample.T * wl.T).T, axis=1)
        xmix = x + xm * (1 - m)
        return l_out_sample, wl, xm, xmix

    def _batch_train(self, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            outdic, q_z, out_sample = self.model(data)
            #in VAE, z is sample from q_z, then obtain the ouput recon_x by decode(z)
            #but in MIWAE, z is sample from p_x_given_z
            if self.model.loss=='MIWAE_ELBO':
                loss = self.MIWAE_ELBO(outdic)
            elif self.model.loss=='VAE_loss':
                indic = {
                    'x': data,
                    'recon_x': out_sample,
                }
                loss = self.VAE_loss(outdic, indic)
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

    def evaluation(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(self.test_loader):
                data = data.to(self.device)
                outdic, q_z, out_sample = self.model(data)
                if self.model.loss=='MIWAE_ELBO':
                    val_loss += self.MIWAE_ELBO(outdic)
                elif self.model.loss=='VAE_loss':
                    indic = {
                        'x': data,
                        'recon_x': out_sample,
                    }
                    val_loss += self.VAE_loss(outdic, indic)

        val_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(val_loss))
    
    def train(self, max_epochs):
        for epoch in range(1, max_epochs + 1):
            self._batch_train(epoch)
            self.evaluation()

class GANtrainer(object):
    def __init__(self, Generator, Discriminator):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gen = Generator
        self.gen.to(self.device)
        self.gen_optimizer = optim.Adam(self.gen.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
        self.dis = Discriminator 
        self.dis.to(self.device)       
        self.dis_optimizer = optim.Adam(self.dis.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    
    def BCE_loss(self, d_indicator, indicator):
        return nn.BCEWithLogitsLoss(reduction="elementwise_mean")(d_indicator, indicator)

    def gen_loss(self, alpha, indicator, d_indicator, gen_x, x):
        G_loss1 = ((1 - indicator) * (torch.sigmoid(d_indicator)+1e-8).log()).mean()/(1-indicator).sum()
        G_mse_loss = nn.MSELoss(reduction="elementwise_mean")(indicator*x, indicator*gen_x) / indicator.sum()
        G_loss = G_loss1 + alpha*G_mse_loss
        return G_loss

    def _train_dis(self, x, z, m, h):
        G_sample = self.gen(x, z, m)
        d_indicator = self.dis(x, m, G_sample, h)
        BCE_loss = self.BCE_loss(d_indicator, m)
        BCE_loss.backward()
        self.dis_optimizer.step()
        self.dis_optimizer.zero_grad()
        return BCE_loss
    
    def _train_gen(self, x, z, m, h, alpha):
        G_sample = self.gen(x, z, m)
        d_indicator = self.dis(x, m, G_sample, h)
        d_indicator.detach_()
        G_loss = self.gen_loss(alpha, m, d_indicator, G_sample, x)
        G_loss.backward()
        self.gen_optimizer.step()
        self.gen_optimizer.zero_grad()
        G_mse = nn.MSELoss(reduction="elementwise_mean")((1-m)*x, (1-m)*G_sample) / (1-m).sum()
        return G_mse
    
    def evaluation(self):
        pass

    def train(self, max_epochs):
        self.train_loader = ...
        for epoch in range(1, max_epochs + 1):
            BCE_loss = self._train_dis(x, z, m, h)
            G_mse = self._train_gen(x, z, m, h, alpha)
            if epoch % 100 == 0:
                print('====> Epoch: {} BCE_loss of discriminator: {:.4f} MSE of generator: {:.4f}'.format(
                    epoch, BCE_loss / len(self.train_loader.dataset), G_mse / len(self.train_loader.dataset)))
            self.evaluation()

        

        