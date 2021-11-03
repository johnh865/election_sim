# -*- coding: utf-8 -*-
"""
Test multi-district concatenation

"""
import pdb
import pytest
import numpy as np

import votesim
from votesim.models import spatial
from votesim import metrics
from votesim.metrics.metrics import multi_district_stats
from votesim.metrics import PrRegret


import matplotlib.pyplot as plt


def test_multi():
    v = spatial.Voters(seed=0)
    v.add_random(100, 2)
    
    results = []
    for ii in range(10):
        c = spatial.Candidates(voters=v, seed=ii)
        c.add_random(10)
        e = spatial.Election(voters=v, candidates=c)
        result = e.run(etype='plurality')
        results.append(result)
    
    estat_list = [r.stats for r in results]
    m = multi_district_stats(estat_list)
    
    pr = PrRegret(m)
    
    cprefs = m.candidates.pref
    winners = m.winner.winners
    
    pdb.set_trace()
    plt.plot(cprefs[winners,0], cprefs[winners,1], '.')
    
    
if __name__ =='__main__':
    test_multi()