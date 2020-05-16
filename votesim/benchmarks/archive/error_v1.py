
import os
import itertools
import pandas as pd
import numpy as np

import votesim
from votesim.models import spatial
from votesim.benchmarks.runtools import benchrun
from votesim.benchmarks.simpleNd import SimpleCreate, CaseGenerator


def model(
           methods, 
           name,
           ndim,
           error_mean,
           error_width,
           clim_mean,
           clim_width,
           seed=1,
           cnum=8,
           trialnum=1000,
           numvoters=100,
           strategy='candidate', 
          ):
    
    # seed = kwargs.get('seed', 1)
    # methods = kwargs['methods']
    # name = kwargs['name']
    # ndim = kwargs['ndim']
    # error_mean = kwargs['error_mean']
    # error_width = kwargs['error_width']
    
    # cnum = kwargs['cnum']
    
    e = spatial.Election(None, None, seed=seed, name=name)
    v = spatial.ErrorVoters(seed=seed, strategy=strategy)
    v.add_random(numvoters=numvoters, 
                 ndim=ndim,
                 error_mean=error_mean,
                 error_width=error_width,
                 clim_mean=clim_mean,
                 clim_width=clim_width,
                 )
    for trial in range(trialnum):
        c = spatial.Candidates(v, seed=trial)
        c.add_random(cnum, sdev=3)
        e.set_models(voters=v, candidates=c)
        e.user_data(
                    strategy=strategy,        
                    num_dimensions=ndim,
                    num_candidates=cnum,
                    error_mean=error_mean,
                    error_width=error_width,
                    )
        
        for method in methods:
            e.run(etype=method)
    return e

                    
    

class ErrorCreate(SimpleCreate):
    
    
    def plot_error_mean(self, 
                         post_file='',
                         plot_file='', 
                         x_axis='error_mean',
                         y_axis='etype',
                         key='output.regret.efficiency_voter',
                         func='subtract100',
                         **kwargs):
        
        return self.plot(post_file,
                         plot_file,
                         x_axis,
                         y_axis,
                         key,
                         func,
                         **kwargs)
    
    
class ErrorDummy:
    name = 'error-dummy'
    kwargs = {}
    kwargs['name'] = name
    kwargs['seed'] = 0
    kwargs['numvoters'] = 10
    kwargs['trialnum'] = 10
    kwargs['ndim'] = (1, 2)
    kwargs['strategy'] = ('candidate', 'voter')
    kwargs['cnum'] = 5
    kwargs['clim_mean'] = -1
    kwargs['clim_width'] = 0,
    kwargs['error_mean'] = [0, .5, 1]
    kwargs['error_width'] = [0, .5, 1]
    
    _model = ErrorCreate(name, model, kwargs)
    run = _model.run
    post = _model.post
    plot = _model.plot_error_mean
    
    
class Error1:
    name = 'error-1'
    kwargs = {}
    kwargs['name'] = name
    kwargs['seed'] = 0
    kwargs['numvoters'] = 100
    kwargs['trialnum'] = 1000
    kwargs['ndim'] = [1, 2,]
    kwargs['strategy'] = ('candidate', 'voter')
    kwargs['cnum'] = 8
    kwargs['clim_mean'] = -1
    kwargs['clim_width'] = 0,
    kwargs['error_mean'] =  [0,  .5,  1.0, 1.5]
    kwargs['error_width'] = [0, 1.0]
    
    _model = ErrorCreate(name, model, kwargs)
    run = _model.run
    post = _model.post
    plot = _model.plot_error_mean
    
    

