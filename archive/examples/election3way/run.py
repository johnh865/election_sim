# -*- coding: utf-8 -*-

"""
Honest election simulations comparing IRV, plurality, and score voting.
Pickle output. 

"""
import itertools

import votesim
from votesim.models import spatial
from votesim.utilities.write import StringTable

import matplotlib.pyplot as plt
#import seaborn as sns

import numpy as np
import pandas as pd


#votesim.logconfig.setInfo()
#votesim.logconfig.setDebug()
votesim.logconfig.setWarning()
outputfile = 'election3way.pkl'
types = ['irv', 'score', 'star', 'plurality', 'smith_minimax', 'approval50', 'approval75']
metric_name = 'stats.regret.vsp'


v = spatial.SimpleVoters(0)
v.add_random(500, ndim=1)
c = spatial.Candidates(v, 0)
e = spatial.Election(v, None, seed=0, scoremax=5)

### Develop parametric study variables
distances = np.linspace(0, 3, 20)[1:]
skew = np.linspace(0,1, 10)[1:-1]
offset = np.linspace(0, 0.5, 10 )

### Loop through parameters
loop_product = itertools.product(distances, skew, offset)
data = []
for ii, (d, s, o) in enumerate(loop_product):
    print(ii, end=',')
    c.reset()
    a = np.array([0, s, 1]) - 0.5
    a = a * d + o
    a = np.atleast_2d(a).T
    c.add(a)
    e.set_data(v, c)
    
    args = {'distance':d, 'skew':s, 'offset':o}
    e.user_data(**args)
    
    data.append(a)
    for t in types:
        e.run(etype=t)
data = np.array(data)
e.save(outputfile)


