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

#import sys
from utils.dataframe import UCIDatasets
from utils.experiment import exp_imputation
from model.MIWAE import MIWAE
from model.imputer import imputer
from utils.dataframe import dataframe, UCIDatasets
from utils.trainer import VAE_trainer, GAN_trainer
from utils.experiment import *
"""
Use the MIWAE and not-MIWAE on UCI data

Find a data from here
https://archive.ics.uci.edu/ml/datasets.php
"""
parser = argparse.ArgumentParser(description='VAE Example')

parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100000, metavar='N',
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

name = args.data
n_hidden = 128
n_samples = 20
max_iter = args.epochs
batch_size = args.batch_size
impute_sample = 10000

###   the missing model   ###
# mprocess = 'linear'
# mprocess = 'selfmasking'
mprocess = 'selfmasking_known'

# ---- number of runs
runs = 1
RMSE_result = dict()
methods = ['miwae','notmiwae','mean','mice','RF']
for method in methods:
    RMSE_result[method] = []
"""
load data: white wine
"""
data = UCIDatasets(name=name)
N, D = data.N, data.D
dl = D - 1
optim_kwargs = {'lr': 0.0001, 'betas': (0.9, 0.999), 'eps': 1e-08 }
MIWAE_kwargs = {
    'data_dim': D, 'z_dim': dl, 'h_dim': n_hidden, 'n_samples': n_samples
    }
notMIWAE_kwargs = {
    'data_dim': D, 'z_dim': dl, 'h_dim': n_hidden, 'n_samples': n_samples, 'missing_process': mprocess
    }
data_kwargs = {
    'batch_size': batch_size
    }
imputer_par = {
    'missing_values': np.nan, 'max_iter': 10, 'random_state': 0, 'n_estimators': 100, 'n_neighbors': 3, 'metric': 'nan_euclidean'
    }
exp_kwargs = {
    'dataset':name, 'runs':runs, 'seed': args.seed,
}
config = {
    'exp_kwargs': exp_kwargs, 'optim_kwargs': optim_kwargs,
    'MIWAE_kwargs': MIWAE_kwargs, 'notMIWAE_kwargs': notMIWAE_kwargs,
    'data_kwargs': data_kwargs, 'imputer_par': imputer_par,
    }

def main():
    RMSE_result = exp_imputation( 'exp_imputation', model_list = ['miwae', 'notmiwae'], config = config, num_of_epoch = max_iter)

    print("RMSE_miwae = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_result['miwae']), np.std(RMSE_result['miwae'])))
    print("RMSE_notmiwae = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_result['notmiwae']), np.std(RMSE_result['notmiwae'])))
    print("RMSE_mean = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_result['mean']), np.std(RMSE_result['mean'])))
    print("RMSE_mice = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_result['mice']), np.std(RMSE_result['mice'])))
    print("RMSE_missForest = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_result['RF']), np.std(RMSE_result['RF'])))

if __name__ == "__main__":
    main()