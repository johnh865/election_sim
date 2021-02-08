# -*- coding: utf-8 -*-

import os
import itertools

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import definitions
import globalcache

# sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

sns.set()
pd.options.mode.chained_assignment = 'raise'

import votesim
from votesim.benchmarks import simple
from votesim import plots, post

# %% Read
benchmark = simple.simple_base_compare_test()

dirname = definitions.DIR_DATA_BENCHMARKS
dirname = os.path.join(dirname, benchmark.name)

@globalcache.cache_decorate('read')
def read():
    return benchmark.read(dirname=dirname)



g = globalcache.create(globals())
p = read()
df = p.dataframe

# %%


tolname = 'args.voter-0.1.set_behavior.tol'
basename = 'args.voter-0.1.set_behavior.base'
yname = 'args.etype'
zname = 'output.winner.regret_efficiency_candidate'


df1 = df[[
    tolname, basename, yname, zname]].copy()
df1[zname] = df1[zname] * 100

groupby = df1.groupby(by=basename)
for basename1 in groupby.groups:
    dfb = groupby.get_group(basename1)
    plt.figure()
    plt.title(basename1)
    plots.heatmap(x=tolname, y=yname, hue=zname, data=dfb)
    