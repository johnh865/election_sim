# -*- coding: utf-8 -*-
import os
import itertools

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import globalcache


# sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

sns.set()


import votesim
from votesim.benchmarks import simple


# %% Load the data

@globalcache.cache_decorate('read')
def read():
    benchmark = simple.simple3way()
    dir_bench = votesim.definitions.DIR_DATA_BENCHMARKS
    dirname = os.path.join(dir_bench, benchmark.name)
    benchmark.read(dirname=dirname)
    return benchmark


g = globalcache.create(globals())
b = read()


# %% Rerun
df = b.reader.dataframe

xname = 'output.winner.regret_efficiency_voter'

regret = (100* (1 - df[xname])).rename('regret')



pratio = (df['output.candidate.plurality_ratio'] * 100).rename('pratio')
vtol = df['args.user.voter_tolerance'].rename('tol')
etype = df['args.etype'].rename('etype')



df1 = pd.concat([regret, pratio, vtol, etype], axis=1) 

# sns.lineplot(x='tol', y='regret', hue='etype', data=df1)



plt.figure()
votesim.plots.heatmap(x='tol', y='etype', hue='regret', data=df1)
plt.subplots_adjust(left=.185, wspace=.025)    
plt.xlabel('Voter Tolerance (voter preference units)')
plt.title('Effect of Voter Tolerance on Voter Regret')
plt.savefig('Regret-vs-Tolerance.png')


df2 = df1.loc[df1['pratio'] < 50.]
plt.figure()
votesim.plots.heatmap(x='tol', y='etype', hue='regret', data=df2)
plt.subplots_adjust(left=.185, wspace=.025)    
plt.xlabel('Voter Tolerance (voter preference units)')
plt.title('Effect of Voter Tolerance on Voter Regret, for PVR < 50%')
plt.savefig('Regret-vs-Tolerance-less50.png')




