import torch
import torch.nn as nn
import torch.distributions as pdfun

"""
GAIN: https://www.vanderschaar-lab.com/papers/ICML_GAIN.pdf (ICML, 2018) 
"""

class Discriminator(nn.Module):
    def __init__(self, data_dim: int):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(data_dim*2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, data_dim)
        self.init_weight()
    
    def init_weight(self):
        layers = [self.fc1, self.fc2, self.fc3]
        [nn.init.xavier_normal_(layer.weight) for layer in layers]
        
    def forward(self, x, m, g, h):
        """
        eq(4) in the paper
        
        input:
        batch self.M, self.X in utils/dataframe
        g: from generator
        h: Hint Matrix
        """ 
        inp = m * x + (1-m) * g 
        inp = torch.cat((inp, h), dim=1)
        out = nn.ReLU(self.fc1(inp))
        out = nn.ReLU(self.fc2(out))
        #out = nn.Sigmoid(self.fc3(out)) # Prob Output
        out = self.fc3(out)
        return out    

class Generator(torch.nn.Module):
    def __init__(self, data_dim):
        super(Generator, self).__init__()
        self.fc1 = torch.nn.Linear(data_dim*2, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, data_dim)
        nn.ReLU()
        nn.Sigmoid()
        self.init_weight()
    
    def init_weight(self):
        layers = [self.fc1, self.fc2, self.fc3]
        [torch.nn.init.xavier_normal_(layer.weight) for layer in layers]
        
    def forward(self, x, z, m):
        """
        eq(2) in the paper
        input:
        batch self.M, self.X in utils/dataframe
        z: from sampling
        """ 
        inp = m * x + (1-m) * z
        inp = torch.cat((inp, m), dim=1)
        out = nn.ReLU(self.fc1(inp))
        out = nn.ReLU(self.fc2(out))
        out = nn.Sigmoid(self.fc3(out)) # Prob Output
        #out = self.fc3(out)
        return out 