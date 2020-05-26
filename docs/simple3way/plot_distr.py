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


votesim.logSettings.start_debug()

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


keys = b.reader.parameters_user
groupby = df.groupby(keys)


key = list(groupby.groups.keys())[0]
# df = groupby.get_group(key)
# df = df.reset_index()
vseed = df['args.voter.0.set_seed.seed']
cseed = df['args.candidate.0.set_seed.seed']

vunique, indices_seed = np.unique(vseed, return_index=True)
cunique, indices_cseed = np.unique(cseed, return_index=True)

# indices = df.sample(1000).index



voter_prefs = []
cand_prefs = []
for ii in indices_seed:
    e = b.rerun(index=ii)
    v = e.voters.voters
    print('seed', e.voters.seed)
    # c = e.candidates.candidates
    voter_prefs.append(v.copy())
    # cand_prefs.append(c.copy())
    

c = e.candidates

cand_prefs = []  
for ii in indices_cseed[0:1000]:
    e = b.rerun(index=ii)
    c = e.candidates.candidates
    if ii % 100 == 0:
        print('c seed', ii)
    cand_prefs.append(c.copy())
    
    
# %% Plot

votesim.plots.vset()

voter_prefs = np.column_stack(voter_prefs)
cand_prefs = np.column_stack(cand_prefs)


bins = np.arange(-4, 4.25, .25)
sns.distplot(voter_prefs, bins=bins, norm_hist=True,
             kde=True, label='voter pref.')
sns.distplot(cand_prefs, bins=bins, norm_hist=True,
             kde=False, label='candidate pref.')
plt.legend()
plt.xlim(-4.5, 4.5)
plt.xticks(np.arange(-4, 5, 1))
plt.title('Probability Density of Voter & Candidate Preferences')
plt.xlabel('Preference')
plt.ylabel('Probability Density')

plt.savefig('3way-pref-distribution.png')


# key = ('simple-three-way', 'smith_score', 100, 3, 1, 'voter', 3.0)
# dfi = groupby.get_group(key)
