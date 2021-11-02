import numpy as np
import torch
import torch.nn as nn
import torch.distributions as pdfun
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image

"""
Find a data from here
https://archive.ics.uci.edu/ml/datasets.php
"""
class notMIWAE(nn.Module):
    """
    Original paper
    https://arxiv.org/pdf/2006.12871.pdf
    """
    def __init__(self, data_dim,
                z_dim=50, h_dim=100, n_samples=1,
                activation=nn.Tanh,
                out_dist='gauss',
                out_activation=None,
                latent_prior = pdfun.normal.Normal(loc=0.0, scale=1.0),
                learnable_imputation=False,
                permutation_invariance=False,
                embedding_size=20,
                code_size=20, 
                missing_process='selfmask',
                # TODO: the input dim should consider this, 
                #    or this should be same as the data_dim
                #I think is the first one    #paper does not mension embedding
                testing=False,
                imp = None #should be a mask with (1,self.d)
                ):
        """
        X, Y should be complete in experiments
        Then generate the missing data
        """
        if out_dist not in ['gauss', 'normal', 'Bernoulli', 't', 't-distribution']:
            raise ValueError("Only allow 'gauss', 'normal', or 'Bernoulli' as out_dist")
        if missing_process not in ['selfmasking', 'selfmasking_known', 'linear', 'nonlinear']:
            raise ValueError("Only allow 'selfmasking', 'selfmasking_known', 'linear' or 'nonlinear' as 'missing_process'")
        super(notMIWAE, self).__init__()
        self.loss = 'notMIWAE_ELBO'
        self.activation = activation
        self.out_activation = out_activation
        self.d = data_dim
        self.n_samples = torch.reshape(torch.tensor([n_samples]), (-1,)) # sample for latent variable 
        self.embedding_size = embedding_size
        self.code_size = code_size
        self.out_dist = out_dist
        self.latent_prior = latent_prior
        self.testing = testing
        self.missing_process = missing_process
        self.learnable_imputation=learnable_imputation
        self.permutation_invariance=permutation_invariance
        if imp is None:
            self.imp = np.zeros((1, self.d))
        else:
            try:
                self.imp = np.array(imp).reshape((1, self.d))
            except ValueError:
                print("imp should be a mask with shape (1, {}) to indicate if imputating.".format(self.d))
        
        if self.permutation_invariance:
            self.enc = nn.Sequential(
                nn.Linear(self.code_size, h_dim),
                nn.Linear(h_dim, h_dim))\
                if self.activation is None else nn.Sequential(
                nn.Linear(self.d, h_dim),
                self.activation(),
                nn.Linear(h_dim, h_dim),
                self.activation())
        else:
            self.enc = nn.Sequential(
                nn.Linear(self.d, h_dim),
                nn.Linear(h_dim, h_dim))\
                if self.activation is None else nn.Sequential(
                nn.Linear(self.d, h_dim),
                self.activation(),
                nn.Linear(h_dim, h_dim),
                self.activation())
        self.fc_mu = nn.Linear(h_dim, z_dim) # Mean vector 
        self.fc_sig = nn.Linear(h_dim, z_dim) # variance vector
        self.fc_dec = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.Linear(h_dim, h_dim)) \
            if self.out_activation is None else nn.Sequential(
            nn.Linear(z_dim, h_dim),
            self.out_activation(),
            nn.Linear(h_dim, h_dim),
            self.out_activation()) #for decoder
        
        self.fc_dec_ber = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.Linear(h_dim, h_dim)) \
            if self.out_activation is None else nn.Sequential(
            nn.Linear(z_dim, h_dim),
            self.out_activation(),
            nn.Linear(h_dim, h_dim),
            self.out_activation()) #for Bernoulli decoder
        
        self.fc_dec_mu_gauss = nn.Linear(h_dim, self.d)
        self.fc_dec_std = nn.Linear(h_dim, self.d)

        self.fc_dec_mu_t = nn.Linear(h_dim, self.d)
        self.fc_dec_log_sigma = nn.Linear(h_dim, self.d)
        self.fc_dec_df = nn.Linear(h_dim, self.d)

        self.fc_dec_logits = nn.Linear(h_dim, self.d) # prob = y + eps

        self.emb = nn.Linear(self.embedding_size + 1,self.code_size) #for embedding
        self.E = Variable(torch.randn(self.d, self.embedding_size))  #for embedding

        self.mis_procs_linear = nn.Linear(self.d, self.d) #for missing_process
        self.mis_procs_nonlinear = nn.Sequential(
            nn.Linear(self.d, h_dim),
            F.tanh(),
            nn.Linear(h_dim, self.d))#for missing_process
        self.W = Variable(torch.randn(1, 1, self.d))  #for missing_process
        self.b = Variable(torch.randn(1, 1, self.d))  #for missing_process

        self.init_weight()

    def init_weight(self):
        #layers = [self.enc, self.fc_mu, self.fc_sig, self.fc_dec, self.fc_dec_ber, self.fc_dec_mu_gauss,
        #    self.fc_dec_std, self.fc_dec_df, self.fc_dec_logits, self.emb]
        #[nn.init.xavier_normal_(layer.weight) for layer in layers]
        layers = [self.fc_dec_mu_t, self.fc_dec_log_sigma, self.fc_dec_df]
        [nn.init.orthogonal_(layer.weight) for layer in layers]

    def encoder(self, x):
        h = self.enc(x) #in self.d, out h_dim
        q_mu = self.fc_mu(h) #in h_dim, out z_dim
        q_log_sig = self.fc_sig(h) #in h_dim, out z_dim
        q_log_sig = torch.clamp(q_log_sig, min=-10., max=10.) #clip
        return q_mu, q_log_sig

    #randomize latent vector
    def reparameterize(self, mu, log_var):
        std = torch.sqrt(torch.exp(log_var))
        q_z = pdfun.normal.Normal(loc = mu, scale=std)
        # q_z.rsample(self.n_samples): shape [n_samples, batch_size, d] 
        return q_z, q_z.rsample(self.n_samples)

    def _gauss_decoder(self, z):
        z = self.fc_dec(z) #in z_dim, out h_dim
        mu = self.fc_dec_mu_gauss(z) if self.out_activation is None else self.out_activation(self.fc_dec_mu(z)) 
        #nn.init.xavier_normal_(mu.weight)
        #in h_dim, out self.d (X.shape from dataframe.__init__())
        std = F.softplus(self.fc_dec_std(z)) #in h_dim, out self.d (X.shape from dataframe.__init__())
        return mu, std
    
    def _t_decoder(self, z):
        z = self.fc_dec(z) #in z_dim, out h_dim
        mu = self.fc_dec_mu_t(z) 
        mu = mu if self.out_activation is None else self.out_activation(mu)
        #in h_dim, out self.d (X.shape from dataframe.__init__())
        log_sigma = self.fc_dec_log_sigma(z) 
        log_sigma = torch.clamp(log_sigma, min=-10., max=10.) #clip      in h_dim, out self.d (X.shape from dataframe.__init__())
        df = self.fc_dec_df(z) #in h_dim, out self.d (X.shape from dataframe.__init__())
        return mu, log_sigma, df
    
    def _bernoulli_decoder(self, z):
        z = self.fc_dec_ber(z) #in z_dim, out h_dim
        logits = self.fc_dec_logits(z) 
        return logits
    
    def _bernoulli_missing_process(self, z):
        if self.missing_process == 'selfmasking':
            logits = - self.W * (z - self.b)

        elif self.missing_process == 'selfmasking_known':
            logits = - F.softplus(self.W) * (z - self.b)

        elif self.missing_process == 'linear':
            logits = self.mis_procs_linear(z)

        elif self.missing_process == 'nonlinear':
            logits = self.mis_procs_nonlinear(z)

        else:
            print("use 'selfmasking', 'selfmasking_known', 'linear' or 'nonlinear' as 'missing_process'")
            logits = None

        # ---- return logits since it goes better with tfp bernoulli
        return logits
    
    def decoder(self, z): 
        """
        this is for sample and reconstruct
        but do not use in forward since 
        z = z.permute(1, 0, 2) and logits = self._bernoulli_decoder(z) will operate twice
        """
        # self.l_z: shape [n_samples, batch_size, d] 
        z = z.permute(1, 0, 2)  # shape [batch_size, n_samples, d]
        if self.out_dist in ['gauss', 'normal']:
            mu, std = self._gauss_decoder(z)
            # p(x|z)
            p_x_given_z = pdfun.normal.Normal(loc=mu, scale=std)
            l_out_sample = p_x_given_z.sample()

        elif self.out_dist in ['t', 't-distribution']:
            mu, log_sig, df = self._t_decoder(z)
            # p(x|z)
            p_x_given_z = pdfun.studentT.StudentT(loc=mu, scale=torch.nn.softplus(log_sig) + 0.0001,
                                                  df=3 + torch.nn.softplus(df))
            l_out_sample = p_x_given_z.sample()

        elif self.out_dist == 'Bernoulli':
            logits = self._bernoulli_decoder(z)
            # p(x|z)
            p_x_given_z = pdfun.bernoulli.Bernoulli(logits=logits)  # (probs=y + self.eps)
            l_out_sample = p_x_given_z.sample()
        return l_out_sample

    def forward(self, x):
        ##########################
        """
        input:
        batch self.M, self.X in utils/dataframe
        n_samples represents sample dim for sampling
        """ 
        x = torch.reshape(x, (-1,self.d))
        m = torch.isnan(x).float().clone()
        #m = torch.reshape(m, (-1,self.d)) #TODO: need to confirm if data itself is correct, shape is correct now
        x = torch.nan_to_num(x, nan = 0)
        if not self.testing and self.learnable_imputation:
            input_tensor = x + (1-m) * self.imp
        elif not self.testing and self.permutation_invariance:
            input_tensor = self.permutation_invariant_embedding(x, m)
        else:
            input_tensor = x

        input_tensor = torch.reshape(input_tensor, (-1,self.d))

        outdic=dict()
        # encoder
        q_mu, q_log_sig = self.encoder(input_tensor)
        outdic['q_mu'], outdic['q_log_sig'] = q_mu, q_log_sig

        # sample latent values
        q_z, l_z = self.reparameterize(q_mu, q_log_sig)

        """
        VAE stucture only need: l_out_sample = self.decoder(l_z)
        but I compute some complicated term for MIWAE_ELBO in the following
        """
        # parameters from decoder
        # l_z: shape [n_samples, batch_size, d] 
        l_z = l_z.permute(1, 0, 2)  # shape [batch_size, n_samples, d]
        if self.out_dist in ['gauss', 'normal']:
            mu, std = self._gauss_decoder(l_z)
            # p(x|z)
            p_x_given_z = pdfun.normal.Normal(loc=mu, scale=std)
            self.l_out_mu = mu
            l_out_sample = p_x_given_z.sample()

        elif self.out_dist in ['t', 't-distribution']:
            mu, log_sig, df = self._t_decoder(l_z)
            # p(x|z)
            p_x_given_z = pdfun.studentT.StudentT(loc=mu, scale=torch.nn.softplus(log_sig) + 0.0001,
                                                  df=3 + torch.nn.softplus(df))
            self.l_out_mu = mu
            l_out_sample = p_x_given_z.sample()

        elif self.out_dist == 'Bernoulli':
            logits = self._bernoulli_decoder(l_z)
            outdic['logits'] = logits
            # p(x|z)
            p_x_given_z = pdfun.bernoulli.Bernoulli(logits=logits)  # (probs=y + self.eps)
            self.l_out_mu = F.sigmoid(logits) # TODO: logits?
            l_out_sample = p_x_given_z.sample()

        # missing process
        # mix x_o with samples of x_m
        l_out_mixed = l_out_sample * torch.unsqueeze(1 - m, 1) + torch.unsqueeze(x * m, 1)
        logits_miss = self._bernoulli_missing_process(l_out_mixed)

        # p(m|x)
        p_s_given_x = pdfun.bernoulli.Bernoulli(logits=logits_miss)  # (probs=m + eps)

        # evaluate m in p(m|x)
        log_p_m_given_x = torch.sum(p_s_given_x.log_prob(torch.unsqueeze(m, 1)), -1)
        # missing process end

        # q_z is from self.reparameterize
        # q_z_expand is after unsqueeze (n_sample dim)
        # can also input q_z into MIWAE_ELBO and compute inside loss function, but I compute here
        q_z_expand = pdfun.normal.Normal(loc=torch.unsqueeze(q_z.loc, 1), scale=torch.unsqueeze(q_z.scale, 1))
        log_q_z_given_x = torch.sum(q_z_expand.log_prob(l_z), -1) #evaluate the z-samples in q(z|x)

        # log_p_x_given_z is depend on outdist 
        # computing here might be more convenient than computing in trainer
        log_p_x_given_z = torch.sum(
                torch.unsqueeze(m, 1) * p_x_given_z.log_prob(torch.unsqueeze(x, 1)), -1)
                #shape [batch, 1, self.d]   [batch, 1, self.d]
        log_p_z = torch.sum(self.latent_prior.log_prob(l_z), -1) #evaluate the z-samples in the prior
        outdic['lpxz'], outdic['lpmz'], outdic['lqzx'], outdic['lpz'] = log_p_x_given_z, log_p_m_given_x, log_q_z_given_x, log_p_z
        return outdic, q_z, l_out_sample 
        # l_out_sample is sample from p_x_given_z in decoder
        # z is sample from q_z, then obtain the ouput l_out_sample by decode(z)

    def permutation_invariant_embedding(self, x, m):
        """
        https://github.com/microsoft/EDDI

        input:  batch self.M, self.X in utils/dataframe
        output: embedding self.g
        """
        self.Es = torch.unsqueeze(m, 2) * torch.unsqueeze(self.E, 0)  # [ _ , self.d, self.embedding_size]
        print("Es", self.Es.shape)
        self.Esx = torch.cat([self.Es, torch.unsqueeze(x, 2)], 2) #[ _ , self.d, self.embedding_size + 1]
        print("Esx", self.Esx.shape)

        self.Esxr = torch.reshape(self.Esx, (-1, self.embedding_size + 1)) 
        # [ _ , self.d, self.embedding_size  +1] --> [ _ * self.d, self.embedding_size]
        print("Esxr", self.Esxr.shape)
        self.h = F.relu(self.emb(self.Esxr))   # [ _ * self.d, self.embedding_size  +1] --> [ _ * self.d, self.code_size]
        print("h", self.h.shape)
        self.hr = torch.reshape(self.h, (-1, self.d, self.code_size)) # [ _ , self.d, self.code_size] --> [ _, self.d, self.code_size]
        print("hr", self.hr.shape)
        self.hz = torch.unsqueeze(m, 2) * self.hr # multiply on the dim self.code_size   [ _, self.d, self.code_size]
        print("hz", self.hz.shape)
        self.g = torch.sum(self.hz, 1)  # sum feature dim  self.d   [ _, self.d, self.code_size]
        print("g", self.g.shape)
        return self.g 