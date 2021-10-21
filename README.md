# A side project _ Variational inference and Missing not at random

 I have an idea from my extending research of Master's thesis, I may try if this idea can work.

 Not finished yet, updating 

 model/MIWAE finished (pytorch)  
 model/imputer finished (missRandom, mice, knn, simpleimputer)  

# I will reproduce some common methods first and make them easier and more convenient to use in this project.
The `Demo.ipynb` file will demonstrate how to use the objects in the project.

# Some works before this.
 The topic of my Master's thesis is developing methods for handling missing values in online user-generated product reviews (UGPR). (I focused on missing X but not Y)
 I tried to develop an imputation model to deal with missing not at random (MNAR) data. I analyzed the observed values themselves and tried to impute data from similar observations. `All variables in X can be MNAR`

 After graduation, it came to my mind that I should try some latent variable ideas.
 Instead of modeling missing pattern as many MNAR methods, I assumed the MNAR itself as a latent variable since we assumed that the dataset must contained MAR and MNAR observations. ` actually the latent variable represent "if an observation follow the MNAR mechanism" `

 In fact, if `all` observations in a dataset are MNAR, the observed data must be completely truncated such that no methods are able to deal with them. In practice, the cause of missing data are complicated. Even though we know what cause missing data, it is likely that most observations follow this reason but still some observations does not. `All observations in a dataset are MNAR` actually is a strong assumption.

 In our assumption, even a complete observation could follow MNAR mechanism since the value may not satisfy the missing condition.
 Note that if an observation does not follow MNAR mechanism, then this observation is MAR.

 I applied MICE and kNNI (hotdeck, since we thought the MNAR observation might be more similar to other MNAR observations) to impute the MAR and MNAR observations. However, the estimation were not significantly better than applied MICE to the whole dataset even though I recognized MNAR precisely. 
 
 I think the problem might be the imputation methods, maybe I should tried to estimate the distribution p(x_observed | MNAR or not) by discarding MNAR observations and sampling the whole dataset from the distribution obtained from partial observations. 

 # IDEA
 I think we might assumed the MNAR itself as a latent variable and develop a method to estimate the distribution of full dataset. The purpose is to estimate the distribution of full dataset without knowing what cause missingness. If this cannot work, estimate the distribution of full dataset from a prior distribution of the MNAR guessing is also great. 

 This idea originally inspired by bandit problem which estimate an unknown probability itself. In addition, Bayesian idea assume a prior distribution to estimate the posterier. Combine these ideas, I think latent variable using variational inference are really suitable to deal with MNAR. 
 I have successfully recognized MNAR observations by modeling the missing mechanism, I think this idea might really work. But maybe need a lot of time to solve some problems. 

 MNAR frequently occurs in medical data and many researchs deal with it through causal inference. Causal inference might provide some tools to predict the potential output based on some missingness analysis. 
 I have no idea which one (Causal inference and latent variable using variational inference) can develop a general framework.
 Need to try it.

 # Model:
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


 # environment:
 I recommend using docker or anaconda.
 
 Personally Recommend:
 1. R is convenient in processing structured data, but is really slow for developing your own methods.
    If using R, using Rcpp and developing the methods by C++ might be great.  
    Then use `apply` family to employ the function from Rcpp.  
    I use R in my thesis and the extending works.  
    I decide to use python since there are a lot of packages with different usage (relative or not relative to machine learning) in python.
 2. Recommend pytorch rather than tensorflow, if really want to use tf, I recommend tf2 rather than tf1. 
    The major reason is that the documentary of pytorch is really better than tensorflow...
    However, keras provide some api to build model very fast. 
    Keras is great for building models quickly.
    
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
check `build_env.txt`

docker run -it --rm -p 8888:8888 -v /project path/:/mount/src -w /mount/src --gpus "device=0" nvcr.io/nvidia/pytorch:21.09-py3
more docker images see https://ngc.nvidia.com/catalog/containers



