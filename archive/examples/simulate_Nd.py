# -*- coding: utf-8 -*-

"""
Honest election simulations comparing IRV, plurality, and score voting

"""
import votesim
import votesim.simulation as sim
from votesim.votesystems import irv, plurality, score
from votesim.utilities.write import StringTable

import matplotlib.pyplot as plt
#import seaborn as sns

import numpy as np
import pandas as pd


#votesim.logconfig.setInfo()
#votesim.logconfig.setDebug()
votesim.logconfig.setWarning()

ndim = 1
nfactions = 10
ntrials = 1
cnum = 16
error_std = 1
size_mean = 45
width_mean = .2
types = ['irv',
         'rrv',
         'smith_minimax',
         'star',
         'approval',
         'ttr',
         'plurality',
        ]

seedstart = 0
#metric_name = 'stats.regret.vse'
metric_name = 'stats.regret.median_accuracy'

e = sim.Election()
for seed in range(ntrials):
    if seed % 10 == 0:
        print('trial %s' % seed, sep=' ')
    e.set_seed(seed)
    e.set_random_voters(ndim=ndim, 
                        nfactions=nfactions,
                        size_mean=size_mean,
                        width_mean=width_mean,
                        error_std=error_std)
    e.generate_candidates(cnum=cnum)
    for etype in types:
        e.run(etype=etype)


df = e.dataframe()
dframes = []
for etype in types:
    dfi = df.loc[lambda x : x['args.election.etype'] == etype]
    dframes.append(dfi)

#df.to_csv('testfile.csv')

plt.figure()
prange = np.arange(0, 100, 1)
varr = [] 
for dfi, etype in zip(dframes, types):
    varri = []
    for p in prange:
        vsei = np.percentile(dfi[metric_name], p)
        varri.append(vsei)
        
    varri = np.array(varri)
    plt.plot(prange, varri, label=etype)
        
    varr.append(varri)
    
varr = np.array(varr)


plt.legend()
plt.xticks(np.arange(0, 110, 10))
plt.xlim(0, 100)
plt.ylim(0, 1.1)
plt.grid()
#plt.ylabel('Voter Satisfaction Efficiency')
plt.ylabel(metric_name)
plt.xlabel('Percentile of Elections')

t = 'Election simulation with %d factions, %d dimensions, %d candidates' % (nfactions, ndim, cnum)
plt.title(t)
plt.savefig(t + '.png')
    