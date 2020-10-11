"""
Test error generation that error magnitude is indeed growing.

print Solution ought to look like:
    
0.0
0.17958457670544686
0.3591691534108937
0.5387537301163405
0.7183383068217875
0.8979228835272343

"""
# -*- coding: utf-8 -*-

import numpy as np
import sys
from votesim.models import spatial, spatialerror


widths = [0, .1, .2, .3, .4, .5]
errors = []
for error_width in widths:
    
    v1 = spatialerror.ErrorVoters(seed=0)
    v1.add_random(1000, 1, error_mean=0.0, error_width=error_width)
        
    c = spatial.Candidates(v1, seed=0)
    c.add_random(6)
    e1 = spatial.Election(v1, c, seed=0)
    dist_error = e1.ballotgen.distances
    
    v2 = spatial.Voters(seed=0)
    v2.add(v1.pref)
    e2 = spatial.Election(v2, c, seed=0)
    dist_true = e2.ballotgen.distances
    
    error = np.sum(np.abs(dist_error - dist_true))/ len(dist_true)
    errors.append(error)
    print(error)
    
errors = np.array(errors)
condition =  np.argsort(errors) == np.arange(len(errors))

assert np.all(condition)
assert np.all(errors[1:] > 0)

assert np.all(np.diff(errors) > 0)