import os
import torch
import pandas as pd
import zipfile
import random
import numpy as np
#import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import logging
import urllib.request
from os import path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn.model_selection import KFold

"""
Find a data from here
https://archive.ics.uci.edu/ml/datasets.php

url formats are not consistant,
http vs https
housing/housing.data vs concrete/compressive/Concrete_Data.xls
"""
datalist = ["housing", "concrete", "energy", "power", "redwine", "whitewine", "yacht"]

_datadict = {"housing": "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
             "concrete": "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls",
             "energy": "http://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",
             "power": "https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip",
             "redwine": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
             "whitewine": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
             "yacht": "http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data"}
class UCIDatasets(object):
    #https://gist.github.com/martinferianc/db7615c85d5a3a71242b4916ea6a14a2
    def __init__(self,  name,  data_path="./data/", n_splits = 10, seed=1, shuffle = False):
        self.datasets = _datadict
        self.data_path = data_path
        self.name = name
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.seed = seed if self.shuffle else None
        self._load_dataset()
    
    def _load_dataset(self):
        if self.name not in self.datasets:
            raise Exception("Not known dataset!")
        if not path.exists(self.data_path+"UCI"):
            os.mkdir(self.data_path+"UCI")

        url = self.datasets[self.name]
        file_name = url.split('/')[-1]
        if not path.exists(self.data_path+"UCI/" + file_name):
            urllib.request.urlretrieve(
                self.datasets[self.name], self.data_path+"UCI/" + file_name)
        data = None
        """
        Different data formats
        zip, csv, data, xlsx, xls,......
        """
        if self.name == "housing":
            data = pd.read_csv(self.data_path+"UCI/" + file_name,
                        header=0, delimiter="\s+").values
            self.data = data[np.random.permutation(np.arange(len(data)))]
        elif self.name == "concrete":
            data = pd.read_excel(self.data_path+"UCI/" + file_name,
                               header=0).values
            self.data = data[np.random.permutation(np.arange(len(data)))]
        elif self.name == "energy":
            data = pd.read_excel(self.data_path+"UCI/" + file_name,
                                 header=0).values
            self.data = data[np.random.permutation(np.arange(len(data)))]
        elif self.name == "power":
            zipfile.ZipFile(self.data_path +"UCI/CCPP.zip").extractall(self.data_path +"UCI/CCPP/")
            data = pd.read_excel(self.data_path+'UCI/CCPP/Folds5x2_pp.xlsx', header=0).values
            np.random.shuffle(data)
            self.data = data
        elif self.name == "redwine":
            data = pd.read_csv(self.data_path + "UCI/" + file_name,
                               header=1, delimiter=';').values
            self.data = data[np.random.permutation(np.arange(len(data)))]
        elif self.name == "whitewine":
            data = pd.read_csv(self.data_path + "UCI/" + file_name,
                               header=1, delimiter=';').values
            self.data = data[np.random.permutation(np.arange(len(data)))]
        elif self.name == "yacht":
            data = pd.read_csv(self.data_path + "UCI/" + file_name,
                               header=1, delimiter='\s+').values
            self.data = data[np.random.permutation(np.arange(len(data)))]
            
        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state = self.seed)
        self.in_dim = data.shape[1] - 1
        self.out_dim = 1
        self.data_splits = kf.split(data)
        self.data_splits = [(idx[0], idx[1]) for idx in self.data_splits]

    def get_split(self, split=-1, load = "train"):
        if split == -1:
            split = 0
        if 0<=split and split<self.n_splits: 
            train_index, test_index = self.data_splits[split]
            x_train, y_train = self.data[train_index,
                                    :self.in_dim], self.data[train_index, self.in_dim:]
            x_test, y_test = self.data[test_index, :self.in_dim], self.data[test_index, self.in_dim:]
            x_means, x_stds = x_train.mean(axis=0), x_train.var(axis=0)**0.5
            y_means, y_stds = y_train.mean(axis=0), y_train.var(axis=0)**0.5
            x_train = (x_train - x_means)/x_stds
            y_train = (y_train - y_means)/y_stds
            x_test = (x_test - x_means)/x_stds
            y_test = (y_test - y_means)/y_stds
            if load == "train":
                inps = torch.from_numpy(x_train).float()
                tgts = torch.from_numpy(y_train).float()
                train_data = torch.utils.data.TensorDataset(inps, tgts)
                return train_data
            elif load == "test":
                inps = torch.from_numpy(x_test).float()
                tgts = torch.from_numpy(y_test).float()
                test_data = torch.utils.data.TensorDataset(inps, tgts)
                return test_data
            #data = UCIDatasets("housing")
            #train = data.get_split( load="train")
            #train_loader = DataLoader(dataset = train,
            #                    batch_size = batch_size,
            #                    shuffle = shuffle,
            #                    num_workers = 2,
            #                    )

    def get_dataloader(self, split=-1, load = "train", batch_size = 16, shuffle = False, num_workers = 2, ):
        data = self.get_split(split = split, load = load)
        return DataLoader(dataset = data,
                        batch_size = batch_size,
                        shuffle = shuffle,
                        num_workers = num_workers)

"""
TODO: missing data estimate the distribution of the whole dataset
split batch lead to bias or not ?
""" 
class dataframe(object):
    def __init__(self, X, Y, Xval = None, Yval = None):
        """
        X, Y should be complete in experiments
        Then generate the missing data
        """
        if (Xval is None or Yval is None) and not (Xval is None and Yval is None):
            raise ValueError("One of Xval and Yval is None, please check the input.")
        
        self.X_source = np.array(X) #TODO: consider if really need to keep complete data
        self.Y_source = np.array(Y)
        self.n, self.d = self.X.shape
        self.X = self.X_source
        self.Y = self.Y_source
        self.M = np.array(np.isnan(self.X), dtype=np.float32) if np.sum(np.isnan(self.X),axis = None)>0 else None
        if self.M is not None:
            self.source_miss = True
        else:
            self.source_miss = False

        if Xval is not None and Yval is not None:
            self.Xval_source = np.array(Xval)
            self.Yval_source = np.array(Yval)
            self.Xval = self.Xval_source
            self.Yval = self.Yval_source
            self.Mval = np.array(np.isnan(self.Xval), dtype=np.float32) if np.sum(np.isnan(self.Xval),axis = None)>0 else None
        else:
            self.Xval_source = None
            self.Yval_source = None
            self.Xval = None
            self.Yval = None
            self.Mval = None
        if self.Mval is not None:
            self.val_source_miss = True
        else:
            self.val_source_miss = False
        
    def amputation(self, MAR_pattern, MNAR_pattern, MNAR_indicator):
        """
        ampute X and Xval
        MAR_pattern may use a logits(sigmoid) function
        Need to think about how to set experiments
        """
        if self.source_miss and self.val_source_miss:
            return
        pass
        
