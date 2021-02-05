# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import globalcache

# sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
sns.set()


import votesim
from votesim.benchmarks.simpleNd import SimpleThreeWay
dirname = votesim.definitions.DIR_DATA_BENCHMARKS
dirname = os.path.join(dirname, SimpleThreeWay.name)


@globalcache.cache_decorate('read')
def read():
    return SimpleThreeWay.read(dirname=dirname)


# Create the data
# rs = np.random.RandomState(1979)
# x = rs.randn(500)
# g = np.tile(list("ABCDEFGHIJ"), 50)
# df = pd.DataFrame(dict(x=x, g=g))
# m = df.g.map(ord)
# df["x"] += m
g = globalcache.create(globals())

p = read()
df = p.dataframe
yname = 'args.etype'
xname = 'output.winner.regret_efficiency_voter'
ynew = 'type'
xnew = 'regret'

# keys = [yname, xname]
# df = df[keys]

has_majority = df['output.candidate.winner_majority'] > -1


df = df.rename(columns={yname : ynew,
                        xname : xnew,
                        })
df = df

df[xnew] = 1-df[xnew]
df['has_majority'] = has_majority
dfs = df.sample(1000)


pnum = len(np.unique(df[ynew]))
# # Use cubehelix to get a custom sequential palette
pal = sns.cubehelix_palette(pnum, rot=-.5, dark=.3)

# # Show each distribution with both violins and points
# sns.violinplot(x='regret', y='type',
#                 # scale='count',
#                 data=df,)



# sns.swarmplot(x="regret", y="type",
#               hue="output.winner_categories.is_majority",
#               palette=["r", "c", "y"], data=dfs)

# g = sns.boxplot(x="regret", y="type",
#                 hue="has_majority",
#               palette=["r", "c", "y"], data=df,
#               whis=2,linewidth=1, fliersize=1,
#               )


sns.boxenplot(x="regret", y="type",
              palette=['r', 'c'],
              #hue = 'has_majority',
              scale="linear", data=df)


ii = df[ynew] == 'plurality'
xx = df['regret'].loc[ii]
sns.distplot(xx)

# plt.grid('on')
# plt.xlim(None, .5)
# g.set_xscale("log")