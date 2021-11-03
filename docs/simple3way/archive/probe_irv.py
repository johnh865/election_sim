# -*- coding: utf-8 -*-

import os
import itertools

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import globalcache
import definitions

sns.set()

import votesim
from votesim.benchmarks import simple
from votesim import plots

benchmark = simple.simple3way()

dirname = definitions.DIR_DATA_BENCHMARKS
dirname = os.path.join(dirname, benchmark.name)


@globalcache.cache_decorate('read')
def read():
    return benchmark.read(dirname=dirname)



# %% Read 



g = globalcache.create(globals())
p = read()
df = p.dataframe

# %% Post 
## Retrieve elections where IRV and top_two results disagree

yname = 'args.etype'
xname = 'output.winner.regret_efficiency_voter'

vseed = 'args.voter.0.set_seed.seed'
cseed = 'args.candidate.0.set_seed.seed'

parameters = list(p.parameters)
parameters.remove(yname)
df1 = df.copy()



# Filter by voting methods and tolerance
method1 = 'irv'
method2 = 'top_two'

i1 = df1[yname] == method1
i2 = df1[yname] == method2 
i3 = df1['args.user.voter_tolerance'] == 3.0
inew = (i1 | i2) & i3
df1 = df1.loc[inew].reset_index(drop=False)
# df1 = df1.set_index(p.parameters)


df2 = df1[[
           'index', 
           vseed,
           cseed,
           yname, 
           xname,
           ]]


df2 = df2.set_index([vseed, cseed, yname])
df2 = df2.unstack(yname)

regrets = 1 - df2[xname]

disagree = regrets.loc[:, 'irv'] != regrets.loc[:, 'top_two']



df2 = df2.loc[disagree]
# %% Re-run

ii = 0
i_irv = df2.values[ii, 0]
i_tt = df2.values[ii, 1]
e1 = benchmark.rerun(index=i_irv, df=df)
e2 = benchmark.rerun(index=i_tt,  df=df)












