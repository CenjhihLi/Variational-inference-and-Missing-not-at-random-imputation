import itertools
import json
import os
import collections
import gc
import torch
import random
import numpy as np
#nest_asyncio.apply()
from functools import partial

def fs_setup(experiment_name, seed, config, project_dir = "."):
    """
    Setup the experiments fold and use config.json to record 
    the parameters of experiment
    This will use in run_experiment
    very useful since the experiment always stop...
    """
    exp_root_dir = os.path.join(project_dir , "experiments", experiment_name)
    config_path = os.path.join(exp_root_dir, 'config.json')

    # get model config
    if config_path.is_file():
        with config_path.open() as f:
            stored_config = json.load(f)
          
            if json.dumps(stored_config, sort_keys=True) != json.dumps(config, sort_keys=True):
                with os.path.join( exp_root_dir,'config_other.json').open(mode='w') as f_other:
                    json.dump(config, f_other, sort_keys=True, indent=2)
                raise Exception('stored config should equal run_experiment\'s parameters')
    else:
        exp_root_dir.mkdir(parents=True, exist_ok=True)
        with config_path.open(mode='w') as f:
            json.dump(config, f, sort_keys=True, indent=2)
          
    experiment_dir = os.path.join(exp_root_dir, 'seed_{}'.format(seed)) 
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir

def run_experiment(experiment_name, seed, config, num_of_epoch):
    """
    read config,
    load data,
    ampute data,
    train model,
    evaluate results
    """
    _ = config

    experiment_dir = fs_setup(experiment_name, seed, {
        "config1": 'config1',
        'config2': 'config2',
    })
    expr_basename = _
    expr_file = '_.npz'
    
    if expr_file.is_file():
        prev_results = np.load(expr_file, allow_pickle=True)
        _ = prev_results['_'].tolist()
        _ = prev_results['_'].tolist()
        history = prev_results['history']
        start_epoch = len(history)
        if start_epoch >= num_of_epoch:
            print('skipping {} (seed={})   start_epoch({}), num_of_epoch({})'.format(expr_basename, seed, start_epoch, num_of_epoch))
            return
    else:
        history = []
        start_epoch = 0
    
