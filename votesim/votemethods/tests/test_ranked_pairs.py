# -*- coding: utf-8 -*-

import numpy as np
import votesim
from votesim.models import spatial
from votesim.votemethods.condorcet import ranked_pairs

def test1():
    v = spatial.Voters(seed=0,)
    v.add_random(50)
    seed = None
    c = spatial.Candidates(voters=v, seed=seed, )
    c.add_random(12)
    
    e = spatial.Election(voters=v, candidates=c, seed=2)
    # e.run(method=ranked_pairs, btype='rank')
    e.run(etype='ranked_pairs')



def test2():
        
    r = [[2, 0, 1, 3],
           [1, 0, 2, 3],
           [2, 0, 1, 3],
           [2, 0, 1, 3],
           [3, 2, 0, 1],
           [2, 0, 1, 3],
           [3, 2, 0, 1],
           [2, 0, 1, 3],
           [1, 0, 3, 2],
           [2, 0, 3, 1],
           [2, 0, 3, 1],
           [2, 0, 3, 1],
           [2, 0, 1, 3],
           [3, 1, 0, 2],
           [2, 0, 1, 3],
           [3, 2, 0, 1],
           [2, 0, 1, 3],
           [2, 0, 1, 3],
           [1, 0, 2, 3],
           [2, 0, 1, 3]]
    
    r = np.array(r)
    
    w,t,o = ranked_pairs(r)
    
    assert len(w) == 0
    assert 0 in t
    assert 2 in t
    assert len(t) == 2