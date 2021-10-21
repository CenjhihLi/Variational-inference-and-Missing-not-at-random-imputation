# Model: not finish, updating
 First, I may reproduce some methods to see some baseline performance from different latent variables and models. 
 MIWAE and notMIWAE are really similar to the idea, but the latent variables are different.
 In addition, these models might help to make a suitable model to achieve the purpose. 

 MIWAE: http://proceedings.mlr.press/v97/mattei19a/mattei19a.pdf (ICML, 2019)

 notMIWAE: https://arxiv.org/pdf/2006.12871.pdf (2021ICLR)


 other methods:

 `from sklearn.impute import SimpleImputer, MissingIndicator, KNNImputer, IterativeImputer`

  MICE: https://www.jstatsoft.org/article/view/v045i03 (jornal of statistical software, 2011, original paper 2000)

 `Impute (Gibbs sampling) then regress, apply different methods to the regression step to deal with different kind of data.`

 MissForest: https://doi.org/10.1093/bioinformatics/btr597 (Bioinformatics, 2012)

 `allow mix type of variables` This is similar to MICE. Replace the regression model to random forest. 

 `from sklearn.ensemble import RandomForestRegressor, IterativeImputer(estimator = RandomForestRegressor)`

 neural networks:   
 NeuMiss: https://arxiv.org/abs/2106.00311 (Arxive, 2021)

 EDDI: https://arxiv.org/pdf/1809.11142.pdf (active learning, VAE,  PMLR, 2019)

 GAIN: https://www.vanderschaar-lab.com/papers/ICML_GAIN.pdf (ICML, 2018)

 for missing not at random:

 https://doi.org/10.1093/biomet/asz054 (Biometrika, 2019) (this is for missing response y)

 Nonparametric Pattern-Mixture Models: https://arxiv.org/pdf/1904.11085.pdf (Arxive, 2019)
