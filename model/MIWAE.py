import numpy as np
import torch
import torch.nn as nn
import torch.distributions as pdfun
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

from data_prepare.dataframe import dataframe

"""
Find a data from here
https://archive.ics.uci.edu/ml/datasets.php
"""
class MIWAE(nn.Module):
    """
    Original paper
    http://proceedings.mlr.press/v97/mattei19a/mattei19a.pdf
    """
    def __init__(self, data_dim,
                z_dim=50, h_dim=100, n_samples=1,
                activation=F.tanh,
                out_dist='gauss',
                out_activation=None,
                learnable_imputation=False,
                permutation_invariance=False,
                embedding_size=20,
                code_size=20,
                testing=False,
                imp = None #should be a mask with (1,self.d)
                ):
        """
        X, Y should be complete in experiments
        Then generate the missing data
        """
        if out_dist not in ['gauss', 'normal', 'Bernoulli', 't', 't-distribution']:
            raise ValueError("use 'gauss', 'normal', or 'Bernoulli' as out_dist")
        super(MIWAE, self).__init__()
        self.activation = activation
        self.out_activation = out_activation
        self.d = data_dim

        self.enc = nn.Sequential(
            nn.Linear(self.n_samples, self.h_dim),
            self.activation,
            nn.Linear(self.h_dim, self.h_dim),
            self.activation)
        self.fc_mu = nn.Linear(self.h_dim, self.z_dim) # Mean vector 
        self.fc_sig = nn.Linear(self.h_dim, self.z_dim) # variance vector
        self.fc_dec = nn.Sequential(
            nn.Linear(self.z_dim, self.h_dim),
            self.out_activation,
            nn.Linear(self.h_dim, self.n_samples),
            self.out_activation) #for decoder
        
        self.fc_dec_ber = nn.Sequential(
            nn.Linear(self.z_dim, self.h_dim),
            self.out_activation,
            nn.Linear(self.h_dim, self.n_samples),
            self.out_activation) #for Bernoulli decoder
        
        self.fc_dec_mu = nn.Linear(self.h_dim, self.d)
        self.fc_dec_std = nn.Linear(self.h_dim, self.d)
        self.fc_dec_log_sigma = nn.Linear(self.h_dim, self.d)
        self.fc_dec_df = nn.Linear(self.h_dim, self.d)
        self.fc_dec_logits = nn.Linear(self.h_dim, self.d) # prob = y + eps
        self.emb = nn.Linear(self.embedding_size + 1,self.code_size)

        self.z_dim = z_dim
        self.h_dim = h_dim
        self.n_samples = torch.tensor(n_samples) # sample for latent variable 
        
        self.out_dist = out_dist
        
        self.embedding_size = embedding_size
        self.code_size = code_size
        self.testing = testing
        self.batch_pointer = 0
        self.learnable_imputation=learnable_imputation
        self.permutation_invariance=permutation_invariance
        if imp is None:
            self.imp = np.zeros((1, self.d))
        else:
            try:
                self.imp = np.array(imp).reshape((1, self.d))
            except ValueError:
                print("imp should be a mask with shape (1, {}) to indicate if imputating.".format(self.d))

    def encoder(self, x):
        h = self.enc(x) #in n_samples, out h_dim
        q_mu = self.fc_mu(h) #in h_dim, out z_dim
        q_log_sig = self.fc_sig(h) #in h_dim, out z_dim
        activate = lambda x: torch.clamp(x, min=-10, max=10)
        q_log_sig = activate(q_log_sig) #clip
        return q_mu, q_log_sig

    #randomize latent vector
    def reparameterize(self, mu, log_var):
        std = torch.sqrt(torch.exp(log_var))
        q_z = pdfun.normal.Normal(loc = mu, scale=std)
        return q_z, q_z.rsample()

    def gauss_decoder(self, z):
        z = self.fc_dec(z) #in z_dim, out h_dim
        mu = self.out_activation(self.fc_dec_mu(z)) #in h_dim, out self.d (X.shape from dataframe.__init__())
        std = F.softplus(self.fc_dec_std(z)) #in h_dim, out self.d (X.shape from dataframe.__init__())
        return mu, std
    
    def t_decoder(self, z):
        z = self.fc_dec(z) #in z_dim, out h_dim
        
        mu = self.fc_dec_mu(z) 
        nn.init.orthogonal_(mu.weights)
        mu = self.out_activation(mu) #in h_dim, out self.d (X.shape from dataframe.__init__())

        log_sigma = self.fc_dec_log_sigma(z) 
        nn.init.orthogonal_(log_sigma.weights)
        activate = lambda x: torch.clamp(x, min=-10, max=10)
        log_sigma = activate(log_sigma) #clip      in h_dim, out self.d (X.shape from dataframe.__init__())

        df = self.fc_dec_df(z) #in h_dim, out self.d (X.shape from dataframe.__init__())
        nn.init.orthogonal_(df.weights)
        return mu, log_sigma, df
    
    def bernoulli_decoder(self, z):
        z = self.fc_dec_ber(z) #in z_dim, out h_dim
        logits = self.fc_dec_logits(z) 
        return logits

    def forward(self, x, m):
        ##########################
        """
        input:
        X 
        M 
        n_samples
        """ 
        if not self.testing:
            if self.learnable_imputation:
                input_tensor = x + (1-m) * self.imp
            elif self.permutation_invariance:
                input_tensor = self.permutation_invariant_embedding(x, m)
        else:
            input_tensor = x
        
        # encoder
        self.q_mu, self.q_log_sig2 = self.encoder(input_tensor)
        # sample latent values
        q_z, self.l_z = self.reparameterize(self.q_mu, self.q_log_sig2)
        # self.l_z: shape [n_samples, batch_size, d] #TODO: need to confirm
        self.l_z = self.l_z.permute(1, 0, 2)  # shape [batch_size, n_samples, d]

        # parameters from decoder
        if self.out_dist in ['gauss', 'normal']:
            mu, std = self.gauss_decoder(self.l_z)
            # p(x|z)
            p_x_given_z = pdfun.normal.Normal(loc=mu, scale=std)

            self.log_p_x_given_z = torch.reduce_sum(
                torch.expand_dims(m, axis=1) * p_x_given_z.log_prob(torch.expand_dims(x, axis=1)), axis=-1)

            self.l_out_mu = mu
            self.l_out_sample = p_x_given_z.sample()

        elif self.out_dist in ['t', 't-distribution']:
            mu, log_sig2, df = self.t_decoder(self.l_z)

            # p(x|z)
            p_x_given_z = pdfun.studentT.StudentT(loc=mu, scale=torch.nn.softplus(log_sig2) + 0.0001,
                                                  df=3 + torch.nn.softplus(df))

            self.log_p_x_given_z = torch.reduce_sum(
                torch.expand_dims(m, axis=1) * p_x_given_z.log_prob(torch.expand_dims(x, axis=1)), axis=-1)

            self.l_out_mu = mu
            self.l_out_sample = p_x_given_z.sample()

        elif self.out_dist == 'Bernoulli':
            logits = self.bernoulli_decoder(self.l_z)

            # p(x|z)
            p_x_given_z = pdfun.bernoulli.Bernoulli(logits=logits)  # (probs=y + self.eps)

            self.log_p_x_given_z = torch.reduce_sum(
                torch.expand_dims(m, axis=1) * p_x_given_z.log_prob(torch.expand_dims(x, axis=1)), axis=-1)

            self.l_out_mu = F.sigmoid(logits) # TODO: logits?
            self.l_out_sample = p_x_given_z.sample()

        q_z2 = pdfun.normal.Normal(loc=torch.expand_dims(q_z.loc, axis=1), scale=torch.expand_dims(q_z.scale, axis=1))
        self.log_q_z_given_x = torch.reduce_sum(q_z2.log_prob(self.l_z), axis=-1) #evaluate the z-samples in q(z|x)

        # ---- evaluate the z-samples in the prior
        prior = pdfun.normal.Normal(loc=0.0, scale=1.0)
        self.log_p_z = torch.sum(prior.log_prob(self.l_z), -1) #evaluate the z-samples in the prior
        outdic = {'lpxz': self.log_p_x_given_z, 'lqzx': self.log_q_z_given_x, 'lpz': self.log_p_z}
        return outdic, q_z2

    def permutation_invariant_embedding(self, x, m):
        """
        https://github.com/microsoft/EDDI

        input:  batch self.M, self.X 
        output: embedding self.g
        """
        E = Variable(torch.randn(self.d, self.embedding_size))
        Es = torch.expand_dims(m, axis=2) * torch.expand_dims(E, axis=0)  # [-1, self.d, self.embedding_size]
        print("Es", Es.shape)
        Esx = torch.cat([Es, torch.expand_dims(x, axis=2)], axis=2) #[ _ , self.d, self.embedding_size + 1]
        print("Esx", Esx.shape)

        Esxr = torch.reshape(Esx, [-1, self.embedding_size + 1]) # [ _ , self.d, self.embedding_size  +1] --> [ _ * self.d, self.embedding_size]
        print("Esxr", Esxr.shape)
        h = F.relu(self.emb(Esxr))   # [ _ , self.d, self.embedding_size  +1] --> [ _ * self.d, self.code_size]
        print("h", h.shape)
        hr = torch.reshape(h, [-1, self.d, self.code_size]) # [ _ , self.d, self.code_size] --> [ _, self.d, self.code_size]
        print("hr", hr.shape)
        hz = torch.expand_dims(m, axis=2) * hr # multiply on the dim self.code_size   [ _, self.d, self.code_size]
        print("hz", hz.shape)
        g = torch.sum(hz, 1)  # multiply on the dim self.code_size   [ _, self.d, self.code_size]
        print("g", g.shape)
        return g 
        