# -*- coding: utf-8 -*-

"""Template for creating a voting benchmark"""

import time
from votesim.benchmarks.runtools import benchrun
from votesim.models import spatial


BENCHMARK_NAME = 'dummy'
OUTPUT_FILE = BENCHMARK_NAME + '-%s.pkl.gz'

def model(x, methods):
    """Define election model here
    
    Parameters
    ----------
    x : tuple
        Input arguments created from generator `case_args`
        
    Returns
    --------
    out : Election
        Election object.
    """
    #time.sleep(1)
    seed = x
    cnum = 2
    vnum = 10
    ndim = 1
    strategy = 'candidate'
    trialnum = 2
    
    e = spatial.Election(None, None, seed=seed, name=BENCHMARK_NAME)
    v = spatial.Voters(seed=seed, strategy=strategy)
    v.add_random(vnum, ndim=ndim)
    
    for trial in range(trialnum):
        c = spatial.Candidates(v, seed=trial)
        c.add_random(cnum, sdev=1.5)
        e.set_models(voters=v, candidates=c)
        e.user_data(seed=seed)
        
        for method in methods:
            e.run(etype=method)
    
    return e


def case_args(methods):
    """Define benchmark parameters in this generator
    
    Parameters
    ----------
    methods : list of str
        Voting methods to evaluate.
        
    Yields
    ---------
    args : tuple
        Arguments passed onto benchmark `model` function. 
    """
    
    for i in range(15):
        yield (i, methods)
        
        
def run(methods, filename=OUTPUT_FILE, cpus=1):
    """Define function to run benchmark"""    
    return benchrun(methods,
                     model=model,
                     case_args=case_args,
                     filename=filename,
                     cpus=cpus,
                     )



    
    
    