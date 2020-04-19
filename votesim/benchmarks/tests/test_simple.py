# -*- coding: utf-8 -*-
"""
Test running simple benchmark and the post processor
"""
from textwrap import wrap


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc 
import seaborn as sns

import votesim
import votesim.benchmarks
import votesim.benchmarks.simpleNd as sn
import votesim.benchmarks.runtools as rt

#############################################


font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 1}
rc('font', **font)
sns.set()

#############################################
methods = ['plurality',
           'irv',
           'smith_minimax']
sn.SimpleDummy.run(methods, filename = 'test-out-%s.pkl.gz')

filename = 'test-out-%s.pkl.gz' % 'plurality'
df = pd.read_pickle(filename)
print(df.head())


p = rt.PostProcessor('test-out-*')

p.parameter_stats('test-groups-mean.pkl.gz')
p.etype_stats('test-stats.pkl.gz')

#####################################################################################
# Read data 
df1 = pd.read_pickle('test-groups-mean.pkl.gz')
df2 = pd.read_pickle('test-stats.pkl.gz')



######################################################
# Retrieve metric to plot
output_name = 'output.regret.efficiency_voter'
stat_name = 'mean'
output_key = (output_name, stat_name)

series = df1[output_key]
regret =( 1 - series) * 100
param_names = regret.index.names
regret = regret.to_frame()





######################################################
### Rename indices
param_names_new = []
for n in param_names:
    new = n.split('.')[-1]
    param_names_new.append(new)
    
regret.index.names = param_names_new

### Retrieve plot axes categories
x_axis = 'num_candidates'
y_axis = 'etype'

param_names2 = list(param_names_new)
param_names2.remove(x_axis)
param_names2.remove(y_axis)


######################################################
# Sort the tables by metric
# df3 = regret.reset_index()
df3 = regret
gb3 = df3.groupby('etype')
df3 = gb3.agg('mean')
isort = np.argsort(df3.values.ravel())
sort_index = df3.index.values[isort]


######################################################
### Plot grouping
groupby = regret.groupby(param_names2)
groupkeys = list(groupby.groups.keys())



pivot_tables = []
for key in groupkeys:
    dfp = groupby.get_group(key)
    dfp = dfp.reset_index()
    dfp = dfp.pivot(y_axis, x_axis, output_key)
    dfp = dfp.loc[sort_index]
    pivot_tables.append(dfp)


# param_names3 = []
# for n in param_names2:
#     new = n.split('.')[-1]
#     param_names3.append(new)
    
for ii, key in enumerate(groupkeys):
    title = ''
    for name, v in zip(param_names2, key):
        s = '%s=%s, ' % (name, v)
        title += s
    print(title)
    title = '\n'.join(wrap(title, 40))
    
    if ii % 2 == 1:
        yticklabels=False
    else:
        yticklabels=True
        f, axes = plt.subplots(nrows=1,
                               ncols=2,
                               figsize=(12, 7))
        axes = axes.ravel()
    ax = axes[ii % 2 ]
    dfp = pivot_tables[ii]
    sns.heatmap(dfp, 
                ax=ax,
                annot=True, 
                fmt=".1f", 
                cbar=False,
                linewidths=.5,
                yticklabels=yticklabels,
                vmin=0,
                vmax=50, 
                cmap='viridis_r',
                )        
    ax.set_title(title)
    ax.set_ylabel('')



