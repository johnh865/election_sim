# -*- coding: utf-8 -*-
import os
import itertools

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import globalcache
import definitions

# sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

sns.set()


import votesim
from votesim.benchmarks import simple
from votesim import plots, post


# %% Load the data

benchmark = simple.simple5dim()
dirname = definitions.DIR_DATA_BENCHMARKS
dirname = os.path.join(dirname, benchmark.name)


@globalcache.cache_decorate('read')
def read():
    return benchmark.read(dirname=dirname)


# %% Read 

zname = 'output.winner.regret_efficiency_voter'
xname = 'args.user.num_dimensions'
yname = 'categories'
name4 =  'args.user.num_candidates'

g = globalcache.create(globals())
p = read()
df = p.post_data
df = post.categorize_condorcet(df)
regret = (100* (1 - df[zname]))
df.loc[:, 'regret'] = regret 
df.loc[:, 'pratio'] = df['output.candidate.plurality_ratio'] * 100



# %% Post



df1 = df.copy()



num_per_dim = np.sum(df[xname] == 1)
def count1(x):
    return len(x) / num_per_dim * 100



df1 = pd.concat([df['regret'], df['pratio'], df[xname], df[yname]], axis=1)
votesim.plots.plot_1set()
votesim.plots.heatmap(x=xname, y=yname, hue='regret', data=df1,
                      func=count1, fmt='.2f')
plt.xlabel('# of Preference Dimensions')
plt.ylabel('Scenario Categories')
plt.title('% Probability of Scenarios vs Model Dimensions')
plt.savefig('scenarios-vs-dimension.png')


# %%% Plot for only 2-dimensional elections

df1 = df.copy()
df1 = df1.loc[df1[xname] == 2]
df1 = pd.concat([df[name4], df['regret'], df['pratio'],
                 df[xname], df[yname]], axis=1)

votesim.plots.plot_1set()
votesim.plots.heatmap(x=name4, y=yname, hue='regret', data=df1,
                      func=count1, fmt='.2f')
plt.xlabel('# of Candidates')
plt.ylabel('Scenario Categories')
plt.title('% Probability of Scenarios vs # of Candidates, for 2-Dim Model')
plt.savefig('scenarios-vs-candidates.png')

