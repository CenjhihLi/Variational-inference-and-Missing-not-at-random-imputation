import numpy as np
import random
"""
Find a data from here
https://archive.ics.uci.edu/ml/datasets.php
"""

class dataframe:
    def __init__(self, X, Y, Xval = None, Yval = None):
        """
        X, Y should be complete in experiments
        Then generate the missing data
        """
        if (Xval is None or Yval is None) and not (Xval is None and Yval is None):
            raise ValueError("One of Xval and Yval is None, please check the input.")
        
        self.X_source = np.array(X) #TODO: consider if really need to keep coomplete data
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
        """
        if self.source_miss and val_source_miss:
            return
        pass
        
