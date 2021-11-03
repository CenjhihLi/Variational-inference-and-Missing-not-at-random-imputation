import random
import torch
import itertools
import collections
import json
import os
import gc
import pathlib
#import sys
import numpy as np
import pandas as pd
from functools import partial
from torch.utils.data import TensorDataset, DataLoader
from model.MIWAE import MIWAE
from model.notMIWAE import notMIWAE
from model.imputer import imputer
from utils.dataframe import UCIDatasets
from utils.trainer import VAE_trainer
from utils.utils import RMSE

def fs_setup(experiment_name, seed, config, project_dir = "."):
    """
    Setup the experiments fold and use config.json to record 
    the parameters of experiment
    This will use in run_experiment
    very useful since the experiment always stop...
    """
    exp_root_dir = pathlib.Path(project_dir) / f'experiments' / experiment_name
    config_path = exp_root_dir / f'config.json'

    # get model config
    if config_path.is_file():
        with config_path.open() as f:
            stored_config = json.load(f)
          
            if json.dumps(stored_config, sort_keys=True) != json.dumps(config, sort_keys=True):
                with (exp_root_dir / f'config_other.json').open(mode='w') as f_other:
                    json.dump(config, f_other, sort_keys=True, indent=2)
                raise Exception('stored config should equal run_experiment\'s parameters')
    else:
        exp_root_dir.mkdir(parents=True, exist_ok=True)
        with config_path.open(mode='w') as f:
            json.dump(config, f, sort_keys=True, indent=2)
          
    #experiment_dir = exp_root_dir/f'seed_{seed}'
    #experiment_dir.mkdir(parents=True, exist_ok=True)
    return exp_root_dir

def check_training_file(expr_file):
    output = dict()
    if expr_file.is_file():
        #prev_results = np.load(expr_file, allow_pickle=True)
        prev_results = np.load(expr_file)
        history = prev_results['history'].tolist()
        start_epoch = len(history) + 1
    else:
        history = []
        start_epoch = 1
    output = {
        'history': history,
        'start_epoch': start_epoch,
    }
    return output

def exp_imputation(experiment_name, model_list, config, num_of_epoch, CUDA_VISIBLE_DEVICES = "0,1"):
    """
    Use the MIWAE and not-MIWAE on UCI data
    """
    #TODO: use class dataframe(object) to process data and amputation
    def amputation(X):
        _, D = X.shape
        Xnan = X.copy()

        # ---- MNAR in D/2 dimensions
        mean = np.mean(Xnan[:, :int(D / 2)], axis=0)
        ix_larger_than_mean = Xnan[:, :int(D / 2)] > mean
        Xnan[:, :int(D / 2)][ix_larger_than_mean] = np.nan

        Xz = Xnan.copy()
        Xz[np.isnan(Xnan)] = 0

        return Xnan, Xz
    exp_kwargs, optim_kwargs, MIWAE_kwargs, notMIWAE_kwargs, data_kwargs, imputer_par = \
        config['exp_kwargs'], config['optim_kwargs'], config['MIWAE_kwargs'], \
        config['notMIWAE_kwargs'], config['data_kwargs'], config['imputer_par']
    dataset, runs, seed = exp_kwargs['dataset'], exp_kwargs['runs'], exp_kwargs['seed']

    #experiment_name = 'exp_imputation'
    experiment_dir = fs_setup(experiment_name, seed, config)
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # ---- number of runs
    RMSE_result = dict()
    methods = ['miwae','notmiwae','mean','mice','RF']
    for method in methods:
        RMSE_result[method] = []
    
    for _ in range(runs):
        """
        load data: white wine
        """
        data = UCIDatasets(name=dataset)
        # standardize and permute data
        data._normalize()
        data._permutation()

        y = data.data[:, -1:] 
        Xtrain = data.data[:, :-1].copy()
        Xval_org = data.data[:, :-1].copy()
        # introduce missing process
        Xnan, Xz = amputation(Xtrain)
        M = np.array(np.isnan(Xnan), dtype=np.float)
        Xval, Xvalz = amputation(Xval_org)
        # create data_loader
        inps = torch.from_numpy(Xnan).float()
        tgts = torch.from_numpy(y).float()
        train_data = TensorDataset(inps, tgts)
        train_loader = DataLoader(dataset = train_data,
                            **data_kwargs,
                            shuffle = False)

        inps = torch.from_numpy(Xval).float()
        test_data = TensorDataset(inps, tgts)
        test_loader = DataLoader(dataset = test_data,
                            **data_kwargs,
                            shuffle = False)

        for expr_basename in model_list:
            expr_file = experiment_dir / f'{expr_basename}.npz'
            check_point = experiment_dir / f'{expr_basename}_ckpt.pth'
            train_file = check_training_file(expr_file)
            history, start_epoch = train_file['history'], train_file['start_epoch']
            if start_epoch >= num_of_epoch:
                print('skipping {} (seed={})   start_epoch({}), num_of_epoch({})'.format(expr_basename, seed, start_epoch, num_of_epoch))
                continue
            del train_file
            gc.collect()

            train_kwargs = {
                'check_point': check_point, 'expr_file': expr_file, 'start_epoch': start_epoch, 'history': history,
                }                   
            
            if expr_basename == 'miwae':
                # ---- fit MIWAE ---- #
                model = MIWAE(**MIWAE_kwargs)
            elif expr_basename == 'notmiwae':
                # ---- fit not-MIWAE---- #
                model = notMIWAE(**notMIWAE_kwargs)

            # training
            trainer = VAE_trainer(model = model, train_loader = train_loader, test_loader = test_loader, 
                                **data_kwargs, **train_kwargs,
                                optim_kwargs = optim_kwargs) 
            trainer.train(max_epochs=num_of_epoch)

            # imputation RMSE
            l_out_sample, wl, xm, xmix = trainer.imputation(Xz, M)
            if expr_basename == 'miwae':
                RMSE_result['miwae'].append(RMSE(Xtrain, xm, M))
            elif expr_basename == 'notmiwae':
                RMSE_result['notmiwae'].append(RMSE(Xtrain, xm, M))

        # ---- mean imputation ---- #
        impO = imputer(Xnan, method='mean')
        impO.train()
        imp = impO.imp
        Xrec = imp.transform(Xnan)
        RMSE_result['mean'].append(RMSE(Xtrain, Xrec, M))

        # ---- mice imputation ---- #
        impO = imputer(Xnan, method='mice')
        impO.train()
        imp = impO.imp
        Xrec = imp.transform(Xnan)
        RMSE_result['mice'].append(RMSE(Xtrain, Xrec, M))

        # ---- missForest imputation ---- #
        impO = imputer(Xnan, method='missForest')
        impO.train()
        imp = impO.imp
        Xrec = imp.transform(Xnan)
        RMSE_result['RF'].append(RMSE(Xtrain, Xrec, M))

        print('RMSE, MIWAE {0:.5f}, notMIWAE {1:.5f}, MEAN {2:.5f}, MICE {3:.5f}, missForest {4:.5f}'
                .format(RMSE_result['miwae'][-1], RMSE_result['notmiwae'][-1], RMSE_result['mean'][-1], RMSE_result['mice'][-1], RMSE_result['RF'][-1]))
        #print('RMSE, MIWAE {0:.5f}, MEAN {2:.5f}, MICE {3:.5f}, missForest {4:.5f}'
        #      .format(RMSE_result['miwae'][-1], RMSE_result['mean'][-1], RMSE_result['mice'][-1], RMSE_result['RF'][-1]))
    return RMSE_result

    
    
