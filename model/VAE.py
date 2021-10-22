import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as pdfun
import torch.utils.data

class VAE(nn.Module):
    def __init__(self, input_dim = 784, h_dim= 400, z_dim = 20):
        super(VAE, self).__init__()
        self.loss = 'VAE_loss'
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.fc1 = nn.Linear(input_dim, self.h_dim)  #enc
        self.fc21 = nn.Linear(self.h_dim, self.z_dim) #mu
        self.fc22 = nn.Linear(self.h_dim, self.z_dim) #logvar
        self.fc3 = nn.Linear(self.z_dim, self.h_dim) #dec from z to hidden
        self.fc4 = nn.Linear(self.h_dim, input_dim)  #dec to original

    def encoder(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def decoder(self, z):
        h3 = F.relu(self.fc3(z))
        #return F.sigmoid(self.fc4(h3))
        return self.fc4(h3)
    
    #randomize latent vector
    def reparameterize(self, mu, q_log_sig):
        std = torch.sqrt(torch.exp(q_log_sig))
        q_z = pdfun.normal.Normal(mu, std)
        return q_z, q_z.rsample()

    def forward(self, x):
        outdic = dict()
        q_mu, q_log_sig = self.encoder(torch.reshape(x, (-1, self.input_dim)))
        outdic['q_mu'], outdic['q_log_sig'] = q_mu, q_log_sig
                # logvar to std
        q_z, z = self.reparameterize(q_mu, q_log_sig)     # create a torch distribu,   sample with reparameterizationtion
        return outdic, q_z, self.decoder(z)   # z is sample from q_z, then obtain the ouput recon_x by decode(z)