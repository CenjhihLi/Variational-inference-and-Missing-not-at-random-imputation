import os
import gc
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

def _reduce_mem_usage(df):
    """ 
    iterate through all the columns of a pandas.dataframe and modify the data type
    to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

class UCIDatasets(object):
    #https://gist.github.com/martinferianc/db7615c85d5a3a71242b4916ea6a14a2
    def __init__(self,  name,  data_path="./data/", normalize = False, permutation = False, 
                n_splits = 10, seed=1, shuffle = False):
        """
        Store data and preprocessing via numpy
        Perhaps pandas will be more convenient?
        TODO: see if pandas is better 
        """
        self.datasets = _datadict
        self.data_path = data_path
        self.name = name
        self._load_dataset()
        self.N = self.data.shape[0]
        self.y_dim = 1
        self.D = self.data.shape[1] - self.y_dim
        self.normalize = False
        self.permutation = False
        if normalize:
            self._normalize()
        if permutation:
            self._permutation()
        self.n_splits = n_splits
        self._split_dataset(n_splits, seed=seed, shuffle = shuffle)
    
    def _load_dataset(self):
        if self.name not in self.datasets:
            raise Exception("Unknown dataset!")
        if not path.exists(self.data_path+"UCI"):
            os.mkdir(self.data_path+"UCI")

        url = self.datasets[self.name]
        file_name = url.split('/')[-1]
        if not path.exists(self.data_path+"UCI/" + file_name):
            urllib.request.urlretrieve(
                self.datasets[self.name], self.data_path+"UCI/" + file_name)
        """
        Different data formats
        zip, csv, data, xlsx, xls,......
        """
        if self.name == "housing":
            self.data = pd.read_csv(self.data_path+"UCI/" + file_name,
                        header=0, delimiter="\s+").values
        elif self.name == "concrete":
            self.data = pd.read_excel(self.data_path+"UCI/" + file_name,
                               header=0).values
        elif self.name == "energy":
            self.data = pd.read_excel(self.data_path+"UCI/" + file_name,
                                 header=0).values
        elif self.name == "power":
            zipfile.ZipFile(self.data_path +"UCI/CCPP.zip").extractall(self.data_path +"UCI/CCPP/")
            self.data = pd.read_excel(self.data_path+'UCI/CCPP/Folds5x2_pp.xlsx', header=0).values
        elif self.name == "redwine":
            self.data = pd.read_csv(self.data_path + "UCI/" + file_name, header=1, delimiter=';').values
        elif self.name == "whitewine":
            self.data = pd.read_csv(self.data_path + "UCI/" + file_name, header=1, delimiter=';').values
        elif self.name == "yacht":
            self.data = pd.read_csv(self.data_path + "UCI/" + file_name, header=1, delimiter='\s+').values
        del url, file_name
        gc.collect()
        
    def _split_dataset(self, n_splits = 10, seed=1, shuffle = False):  
        seed = seed if shuffle else None      
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state = seed)
        self.data_splits = kf.split(self.data)
        self.data_splits = [(idx[0], idx[1]) for idx in self.data_splits]
        del kf
        gc.collect()
    
    def _normalize(self):  
        if self.normalize:
            print ("The data has already been normalized")
            return 
        else:
            self.data = self.data - np.mean(self.data, axis=0)
            self.data = self.data / np.std(self.data, axis=0)
            self.normalize=True
    
    def _permutation(self):  
        if self.permutation:
            print ("The data has already been permuted")
            return 
        else:
            self.data = self.data[np.random.permutation(self.N)]
            self.permutation=True
    
    def get_dataloader(self, batch_size = 16, shuffle = False, num_workers = 2):
        return DataLoader(dataset = self.data,
                        batch_size = batch_size,
                        shuffle = shuffle,
                        num_workers = num_workers)

    def get_split_dataloader(self, split=-1, load = "train", batch_size = 16, shuffle = False, num_workers = 2, ):
        data = self.get_split(split = split, load = load)
        return DataLoader(dataset = data,
                        batch_size = batch_size,
                        shuffle = shuffle,
                        num_workers = num_workers)

    def get_split(self, split=-1, load = "train"):
        if split == -1:
            split = 0
        if 0<=split and split<self.n_splits: 
            train_index, test_index = self.data_splits[split]
            x_train, y_train = self.data[train_index, :self.D], self.data[train_index, self.D:]
            x_test, y_test = self.data[test_index, :self.D], self.data[test_index, self.D:]
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
        
