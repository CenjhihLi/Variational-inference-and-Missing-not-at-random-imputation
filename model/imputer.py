import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor

method_list = ['mean', 'median', 'most_frequent', 'mice', 'missForest', 'knn']

class imputer(object):
    def __init__(self, X, method):
        self.X = X
        self.method = method
        self._pardict = { 
            'SimpleImputer':{
                'missing_values':np.nan,
                },
            'mice':{
                'max_iter': 10,
                'random_state': 0,
                },
            'missForest':{
                'n_estimators': 100,
                },
            'knn':{
                'n_neighbors':3 ,
                'metric': 'nan_euclidean',
                },            
            }
        self.imp = None
        self._parmap = dict()
        for key in self._pardict:
            value = self._pardict[key]
            for subkey in value:
                self._parmap[subkey] = key
                # might duplicated ?
        #TODO: find some way to use single loop
        del subkey, key, value

    def train(self):
        if self.method in ['mean', 'median', 'most_frequent']:
            #self.imp = SimpleImputer(missing_values=np.nan, strategy=self.method)
            self.imp = SimpleImputer(**self._pardict['SimpleImputer'], strategy=self.method)
        elif self.method == 'mice': 
            self.imp = IterativeImputer(**self._pardict['mice'])
        elif self.method == 'missForest':
            estimator = RandomForestRegressor(**self._pardict['missForest'])
            self.imp = IterativeImputer(estimator=estimator)
        elif self.method == 'knn':
            self.imp = KNNImputer(**self._pardict['knn'])
        self.imp.fit(self.X)
        #for Xtrain without nan and M is missing indicator
        #Xrec = self.imp.transform(X)
        #np.sqrt(np.sum((Xtrain - Xrec) ** 2 * (1 - M)) / np.sum(1 - M))) 
    
    def getParlist(self):
        return self._parmap

    def getParvalue(self):
        out = dict()
        for _ in self._pardict:
            for key in self._pardict[_]:
                out[key] = self._pardict[_][key]
        #TODO: find some way to use single loop
        return out
    
    def par_setting(self, par):
        for key in par:
            value = par[key]
            if key in self._parmap:
                self._pardict[self._parmap[key]][key]=value
            else:
                print('{} is not a parameter.'.format(key))
