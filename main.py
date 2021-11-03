import random
import torch
import numpy as np
import argparse
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torchvision import transforms
from torchvision.utils import save_image

from model.MIWAE import MIWAE
from model.imputer import imputer
from utils.dataframe import dataframe, UCIDatasets
from utils.trainer import trainer
from utils.experiment import *
"""
Find a data from here
https://archive.ics.uci.edu/ml/datasets.php
"""
parser = argparse.ArgumentParser(description='VAE Example')

parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--data', type=str, default='whitewine', metavar='N',
                    help='which dataset from UCI would you like to use?')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

def main():
#    for epoch in range(1, args.epochs + 1):
#        train(epoch)
#        test(epoch)
#        with torch.no_grad():
#            sample = torch.randn(64, 20).to(device)
#            sample = model.decode(sample).cpu()
    pass

if __name__ == "__main__":
    main()