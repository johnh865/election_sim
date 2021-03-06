# -*- coding: utf-8 -*-
import numpy as np

import sys
import votesim
from votesim.votesystems import tools
from votesim.models import vcalcs, spatial
import seaborn as sns
import matplotlib.pyplot as plt

### Create two simulations using 2 strategies
voternum = 1000
candidates = [-0.3, 0.1, 0.4]
candidates = np.atleast_2d(candidates).T


v1 = spatial.SimpleVoters(seed=0, strategy='candidate')
v1.add_random(voternum)
c1 = spatial.Candidates(voters=v1, seed=0)
c1.add(candidates)
v1.calc_ratings(c1.candidates)


v2 = spatial.SimpleVoters(seed=0, strategy='voter')
v2.add_random(voternum)
c2 = spatial.Candidates(voters=v2, seed=0)
c1.add(candidates)
v2.calc_ratings(c1.candidates)


hist, bins = np.histogram(v1.voters, bins=20, density=True)
bins = (bins[0:-1] + bins[1:])/2.
sns.set_style('darkgrid')
for i in range(3):
    
    d1 = v1.voters.ravel()
    r1 = v1.ratings.T[i]
    
    d2 = v2.voters.ravel()
    r2 = v2.ratings.T[i]
    
    plt.subplot(3, 1, i+1)
    sns.lineplot(bins, hist, label='voter distr.', color='k')
    sns.scatterplot(d1, r1, marker='p', s=50, label='strat candidate',)
    sns.scatterplot(d2, r2, marker='.', s=30, label='strat voter', )
    sns.scatterplot([candidates[i]], [.5], marker='o', label='candidate loc.',)
    plt.legend()
    plt.ylabel('voter rating')
    
    
plt.xlabel('voter preference')



for i in range(3):
    
    p = figure(width=500, 
           plot_height=250, 
           title='Ratings vs Regret for Candidate %s' % cnum)  
    d1 = v1.distances.T[i].ravel()
    d2 = v2.distances.T[i].ravel()


