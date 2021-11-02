import torch
import torch.nn.functional as F
import numpy as np

def RMSE(X_train, X_imp, M):
    """
    X_train: original data
    X_imp: imputation
    M: missing indicator matrix
    """
    return np.sqrt(np.sum((X_train - X_imp) ** 2 * (1 - M)) / np.sum(1 - M))
