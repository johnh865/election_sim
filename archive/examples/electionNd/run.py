# -*- coding: utf-8 -*-

"""
This script runs a simple voter simulation for a variety of parameters.

 
"""

import itertools

import votesim
import numpy as np
votesim.logconfig.setWarning()

from votesim.models import spatial
seed = 0
numvoters = 100
trialnum = 1000

ndims = np.arange(1, 6)
strategies = ['candidate', 'voter']

#methods = votesim.votesystems.all_methods.keys()
methods = ['plurality', 'irv', 'score5', 'star5', 'star10', 
           'smith_minimax', 'approval50', 'approval75', 'ranked_pairs']

filename = 'electionNd_data.pkl.zip'

cnums = np.arange(2, 8)


iters = itertools.product(strategies, ndims, cnums)
e = spatial.Election(None, None, seed=seed)
itercount = 0
for (strategy, ndim, cnum) in iters:

    v = spatial.SimpleVoters(seed=seed, 
                             strategy=strategy)
    v.add_random(numvoters, ndim=ndim)
    
    itercount += 1
    print(itercount)
    
    for trial in range(trialnum):
        c = spatial.Candidates(v, seed=trial)
        c.add_random(cnum, sdev=1.5)
        c.add_random(cnum, sdev=3)
        e.set_models(voters=v, candidates=c)
        
        for method in methods:
            e.run(etype=method)
print('building dataframe')
df = e.dataframe()        
print('pickling')
df.to_pickle(filename)       

