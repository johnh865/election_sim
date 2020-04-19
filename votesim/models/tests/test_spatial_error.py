"""
Test error generation that error magnitude is indeed growing
"""
# -*- coding: utf-8 -*-

import numpy as np
import sys
from votesim.models import spatial


widths = [0, .1, .2, .3, .4, .5]
errors = []
for error_width in widths:
    
    v1 = spatial.ErrorVoters(seed=0)
    v1.add_random(1000, 1, error_mean=0.0, error_width=error_width)
        
    c = spatial.Candidates(v1, seed=0)
    c.add_random(6)
    e1 = spatial.Election(v1, c, seed=0)
    dist_error = v1.distances
    
    v2 = spatial.SimpleVoters(seed=0)
    v2.add(v1.voters)
    e2 = spatial.Election(v2, c, seed=0)
    dist_true = v2.distances
    
    error = np.sum(np.abs(dist_error - dist_true))/ len(dist_true)
    errors.append(error)
    print(error)
    
                
condition =  np.argsort(errors) == np.arange(len(errors))

assert np.all(condition)