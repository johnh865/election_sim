# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 18:22:10 2020

@author: John
"""

import os
import itertools

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import globalcache


sns.set()
pd.options.mode.chained_assignment = 'raise'

import votesim
from votesim.benchmarks import tactical
from votesim import plots

benchmark = tactical.tactical0()

dirname = votesim.definitions.DIR_DATA_BENCHMARKS
dirname = os.path.join(dirname, benchmark.name)


# %% Initialize global cache
@globalcache.cache_decorate('read')
def read():
    return benchmark.read(dirname=dirname)

g = globalcache.create(globals())
p = read()
df = p.post_data


# %% Post


param1 = 'args.etype'
param2 = 'args.user.strat_id'
param3 = 'args.user.onesided'

# filter out honest elections
is_honest = df['args.user.strat_id'] == -1
# df = df.loc[~is_honest]



otype = 'regret-voter'
xname1 = 'output.winner.regret_efficiency_voter'
xname2 = 'output.winner.regret_efficiency_candidate'
regret1 = 100 * (1 - df[xname1])
regret2 = 100 * (1 - df[xname2])
regret = np.maximum(regret1, regret2)

pratio = df['output.candidate.plurality_ratio'] * 100
eid = df['args.user.eid']
eid_unique = np.unique(eid)

groupby = df.groupby(by=[param1, param2, param3])
df2 = groupby.agg('mean')
    
    # for eid1 in eid_unique:
#     df1 = df.loc[eid == eid1]
    
    

