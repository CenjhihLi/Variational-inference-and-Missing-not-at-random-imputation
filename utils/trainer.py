import numpy as np
import os
import gc
import pathlib
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

class VAE_trainer(object):
    def __init__(self, model, train_loader, test_loader, batch_size: int = 16, log_interval: int = 10, 
                check_point = './experiments/Demo/VAE_ckpt.pth', 
                expr_file = './experiments/Demo/VAE.npz', 
                start_epoch: int = 1, history = None, 
                optim_kwargs: dict = {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08 }):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.eps = torch.finfo(float).eps #a small epsilon value
        self.model = model #input a model after initialize
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), **optim_kwargs)

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.check_point = check_point
        self.expr_file = expr_file
        self.start_epoch = start_epoch
        self.history = history if history is not None else []

        self.criterion = nn.CrossEntropyLoss()
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        if self.start_epoch>1:
            checkpoint = torch.load(self.check_point)   
            self.model.load_state_dict(checkpoint['net'])   
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def model_summary(self, forward_input: int = 1, dim: int = 10):
        if forward_input == 1:
            summary(self.model, input_size = (16,dim))
        elif forward_input == 2:
            summary(self.model, input_size = [(16,dim),(16,dim)])
        elif forward_input == 3:
            summary(self.model, input_size = [(16,dim),(16,dim),(16,dim)])

    def MIWAE_ELBO(self, outdic: dict, indic = None):
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

    def notMIWAE_ELBO(self, outdic: dict, indic = None):
        # the not-MIWAE ELBO 
        lpxz, lpmz, lqzx, lpz = outdic['lpxz'], outdic['lpmz'], outdic['lqzx'], outdic['lpz'] 
        n_samples = torch.tensor([lpxz.shape[1]])
        l_w = lpxz + lpmz + lpz - lqzx # importance weights in the paper eq(7)(8) 
        log_sum_w = torch.logsumexp(l_w, dim=1)
        log_avg_weight = log_sum_w - torch.log(x = n_samples.type(torch.FloatTensor))
        # ---- average over minibatch to get the average llh
        return torch.sum(log_avg_weight, -1)

    def gauss_loss(self, outdic: dict, indic: dict):
        x = indic['x']
        m = np.array(np.isnan(x), dtype=np.float32)
        mu, log_sig = outdic['q_mu'], outdic['q_log_sig']
        #p(x | z) with Gauss z
        p_x_given_z = - (np.log(2 * np.pi) + log_sig + torch.square(x - mu) / (torch.exp(log_sig) + self.eps))/2.
        return torch.sum(p_x_given_z * m, -1)  # sum over d-dimension 

    def bernoulli_loss(self, outdic: dict, indic: dict):
        x = indic['x']
        m = np.array(np.isnan(x), dtype=np.float32)
        y = outdic['logits']
        #p(x | z) with bernoulli z
        p_x_given_z = x * torch.log(y + self.eps) + (1 - x) * torch.log(1 - y + self.eps)
        return torch.sum(m * p_x_given_z, -1)  # sum over d-dimension
        
    def KL_loss(self, outdic: dict, indic = None):
        q_mu, q_log_sig = outdic['q_mu'], outdic['q_log_sig']
        KL = 1 + q_log_sig - torch.square(q_mu) - torch.exp(q_log_sig)
        return - torch.sum(KL, 1)/2.
    
    def VAE_loss(self, outdic: dict, indic: dict):
        recon_x, x = indic['recon_x'], indic['x']
        q_mu, q_log_sig = outdic['q_mu'], outdic['q_log_sig']
        # BCE = -wi (yi logxi + (1-yi)log(1-xi) )
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        # KLD = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + q_log_sig - torch.square(q_mu) - torch.exp(q_log_sig))
        return BCE + KLD
    
    def imputation(self, x, m, n_samples: int = 0):
        """
        x: Xz[np.isnan(Xnan)] = 0
        """
        outdic, q_z, l_out_sample  = self.model(x, n_samples)
        lpxz, lqzx, lpz = outdic['lpxz'], outdic['lqzx'], outdic['lpz'] 
        l_w = lpxz + lpz - lqzx # importance weights in the paper eq(4) 
        wl = F.softmax(l_w, dim = 1) #TODO: check
        xm = np.sum((l_out_sample.T * wl.T).T, axis=1)
        xmix = x + xm * (1 - m)
        return l_out_sample, wl, xm, xmix

    def _batch_train(self, epoch: int):
        self.model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            outdic, q_z, out_sample = self.model(data)
            #in VAE, z is sample from q_z, then obtain the ouput recon_x by decode(z)
            #but in MIWAE, z is sample from p_x_given_z
            if self.model.loss=='MIWAE_ELBO':
                loss = - self.MIWAE_ELBO(outdic)
            elif self.model.loss=='notMIWAE_ELBO':
                if self.model.testing:
                    loss = - self.MIWAE_ELBO(outdic)
                else:
                    loss = - self.notMIWAE_ELBO(outdic)
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

        train_loss /= len(self.train_loader.dataset)    
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss))
        return train_loss 

    def evaluation(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(self.test_loader):
                data = data.to(self.device)
                outdic, q_z, out_sample = self.model(data)
                if self.model.loss=='MIWAE_ELBO':
                    val_loss += self.MIWAE_ELBO(outdic)
                elif self.model.loss=='notMIWAE_ELBO':
                    if self.model.testing:
                        val_loss = - self.MIWAE_ELBO(outdic)
                    else:
                        val_loss = - self.notMIWAE_ELBO(outdic)
                elif self.model.loss=='VAE_loss':
                    indic = {
                        'x': data,
                        'recon_x': out_sample,
                    }
                    val_loss += self.VAE_loss(outdic, indic)

        val_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(val_loss))
        return val_loss
    
    def train(self, max_epochs: int):        
        for epoch in range(self.start_epoch, max_epochs + 1):
            train_loss = self._batch_train(epoch)
            val_loss = self.evaluation()
            self.history.append([train_loss.item(), val_loss.item()])
            if epoch % 50 == 0:
                """
                save file:
                history-> [loss, accuracy] 
                checkpoint-> model
                """
                checkpoint = {
                    'net': self.model.state_dict(),
                    'optimizer':self.optimizer.state_dict(),
                    }
                torch.save(checkpoint, self.check_point)
                np.savez(self.expr_file,  history=self.history)

class GAN_trainer(object):
    def __init__(self, Generator, Discriminator,
                train_loader, test_loader, batch_size: int = 16, log_interval: int = 10, 
                check_point = './experiments/exp_imputation/GAIN_ckpt.pth', 
                expr_file = './experiments/exp_imputation/GAIN.npz', 
                start_epoch: int = 1, history = None, 
                optim_kwargs: dict = {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08 },
                train_par: dict = {'p_hint':0.9, 'alpha': 10, 'eps': 1e-08}):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gen = Generator
        self.gen.to(self.device)
        self.gen_optimizer = optim.Adam(self.gen.parameters(), **optim_kwargs)
        self.dis = Discriminator 
        self.dis.to(self.device)       
        self.dis_optimizer = optim.Adam(self.dis.parameters(), **optim_kwargs)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.train_par = train_par
        self.check_point = check_point
        self.expr_file = expr_file
        self.start_epoch = start_epoch
        self.history = history if history is not None else []
    
    # Mask Vector and Hint Vector Generation
    def sample_mask(dim_row: int, dim_col: int, prob: float):
        mask_dist = pdfun.uniform.Uniform(0., 1.)
        sample_matrix = mask_dist.rsample(sample_shape = [dim_row, dim_col])
        return (sample_matrix > prob).float() 
    
    # Random sample generator for Z
    def sample_z(dim_row: int, dim_col: int):
        z_dist = pdfun.uniform.Uniform(0., 1.)
        return z_dist.rsample(sample_shape = [dim_row, dim_col])
    
    def BCE_loss(self, d_indicator, indicator):
        return nn.BCEWithLogitsLoss(reduction="elementwise_mean")(d_indicator, indicator)
    
    def Gen_mse_loss(self, x, imputation, indicator):
        return nn.MSELoss(reduction = "elementwise_mean")(indicator * x, indicator * imputation) / indicator.sum()
    
    def mse_loss(self, x, imputation, m):
        return nn.MSELoss(reduction="elementwise_mean")((1-m)*x, (1-m)*imputation) / (1-m).sum()

    def _train_dis(self, x, m, h, G_sample):
        d_indicator = self.dis(x, m, G_sample, h)
        BCE_loss = self.BCE_loss(d_indicator, m)
        BCE_loss.backward()
        self.dis_optimizer.step()
        self.dis_optimizer.zero_grad()
        return BCE_loss
    
    def _train_gen(self, x, m, h, G_sample, alpha: float):
        d_indicator = self.dis(x, m, G_sample, h)
        d_indicator.detach_()
        G_loss1 = ((1 - m) * (torch.sigmoid(d_indicator) + self.train_par['eps']).log()).mean()/(1-m).sum()
        G_mse_loss = self.Gen_mse_loss(self, x, G_sample, m)
        G_loss = G_loss1 + alpha*G_mse_loss
        G_loss.backward()
        self.gen_optimizer.step()
        self.gen_optimizer.zero_grad()
        return G_mse_loss

    def _batch_train(self, epoch: int):
        train_gen_loss = 0
        train_dis_loss = 0
        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device)
            data_dim = data[0].shape[-1]
            z = self.sample_z(self.batch_size, data_dim) 
            x = torch.reshape(data, (-1, data_dim))
            m = torch.isnan(x).float().clone() #missing indicator
            x = torch.nan_to_num(x, nan = 0)
    
            h = self.sample_mask(self.batch_size, data_dim, 1-self.train_par['p_hint'])
            h = m * h + 0.5*(1-h)
    
            z = m * x + (1-m) * z  # Missing Data Introduce
            G_sample = self.gen(x, z, m)
            BCE_loss = self._train_dis(x, z, m, h, G_sample)
            G_mse = self._train_gen(x, z, m, h, G_sample, self.train_par['alpha'])   
            train_dis_loss += BCE_loss         
            train_gen_loss += G_mse

            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    train_gen_loss.item() / len(data)))
        train_dis_loss /= len(self.train_loader.dataset) 
        train_gen_loss /= len(self.train_loader.dataset) 
        return (train_gen_loss, train_dis_loss)

    def evaluation(self):
        self.gen.eval()
        val_loss = 0
        with torch.no_grad():
            for _, (data, _) in enumerate(self.test_loader):
                data = data.to(self.device)
                data = data.to(self.device)
                data_dim = data[0].shape[-1]
                z = self.sample_z(self.batch_size, data_dim) 
                x = torch.reshape(data, (-1, data_dim))
                m = torch.isnan(x).float().clone() #missing indicator
                x = torch.nan_to_num(x, nan = 0)
    
                h = self.sample_mask(self.batch_size, data_dim, 1-self.train_par['p_hint'])
                h = m * h + 0.5*(1-h)
    
                z = m * x + (1-m) * z  # Missing Data Introduce
                G_sample = self.gen(x, z, m)
                val_loss += self.mse_loss(data, G_sample, m)   
        val_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(val_loss))
        return val_loss     

    def train(self, max_epochs: int):
        self.train_loader = ...
        for epoch in range(1, max_epochs + 1):
            (G_mse, BCE_loss) = self._batch_train(epoch)
            val_loss  = self.evaluation()
            #self.history.append([train_loss.item(), val_loss.item()])
            if epoch % self.log_interval == 0:
                print('====> Epoch: {}.  \
                    BCE_loss of discriminator: {:.4f}.  \
                    MSE of generator: {:.4f}.  \
                    Test_loss: {:.4}.'.format(
                    epoch, BCE_loss / len(self.train_loader.dataset), 
                    G_mse / len(self.train_loader.dataset), 
                    val_loss))
            if epoch % 50 == 0:
                """
                save file:
                history-> [loss, accuracy] 
                checkpoint-> model
                """
                checkpoint = {
                    'gen_net': self.gen.state_dict(),
                    'gen_optimizer':self.gen_optimizer.state_dict(),
                    'dis_net': self.dis.state_dict(),
                    'dis_optimizer':self.dis_optimizer.state_dict(),
                    }
                torch.save(checkpoint, self.check_point)
                np.savez(self.expr_file,  history=self.history)
