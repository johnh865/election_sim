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
pd.options.mode.chained_assignment = 'raise'

import votesim
from votesim.benchmarks import simple
from votesim import plots, post

benchmark = simple.simple5dim()

dirname = votesim.definitions.DIR_DATA_BENCHMARKS
dirname = os.path.join(dirname, benchmark.name)



@globalcache.cache_decorate('read')
def read():
    return benchmark.read(dirname=dirname)



    
# %% Read 



g = globalcache.create(globals())
p = read()
df = p.post_data

###########################################
# %% Post


yname = 'args.etype'

# otype = 'regret-efficiency'
# xname = 'output.winner.regret_efficiency_candidate'

otype = 'regret-voter'
xname = 'output.winner.regret_efficiency_candidate'

no_majority = df['output.candidate.winner_majority'] == -1
no_condorcet = df['output.candidate.winner_condorcet'] == -1
regret = 100* (1 - df[xname])
pratio = df['output.candidate.plurality_ratio'] * 100

df = df.reset_index()
df.loc[:, 'plurality_ratio'] = pratio
df.loc[:, 'no_majority'] = no_majority
df.loc[:, 'no_condorcet'] = no_condorcet
df.loc[:, 'regret'] = regret


### Categorize scenario parameters
argname = 'args.user.num_dimensions'
arg_ndim = df[argname]
groupby = df.groupby(by=argname)
keys = groupby.groups.keys()
dframes = (groupby.get_group(k) for k in keys)




# %% Plot categories
### Plot election categories
df = groupby.get_group(list(keys)[0])
etype_num = len(df['args.etype'].unique())


plots.vset()



# %% Plot heatmaps
i = 0
for key, df in zip(keys, dframes):
    sim_num = len(df) / etype_num
    pratio = df['output.candidate.plurality_ratio'] * 100

    plots.subplot_2row()
    plt.subplot(2, 1, 1)
    sns.distplot(pratio, bins=10, norm_hist=True, kde=False)
    plt.xlabel('% plurality winner ratio')
    plt.ylabel('Scenario probability density')
    plt.title('Probability of Plurality Ratio in Benchmark')
    
    plt.subplot(2, 1, 2)
    df = post.categorize_condorcet(df)
    c = df['categories']
    counts = c.value_counts() / len(c)*100
    # sns.barplot(x=counts.keys(), y=counts.values)
    ax = plots.bar(x=counts.keys(), y=counts.values, fmt='g')
    plt.ylabel('% Occurrence')
    # sns.countplot(x='categories', data=df,)
    plt.xlabel('Scenario Categories')
    plt.title('Majority/Condorcet/Utility/Plurality Occurrences')
    
    
    string = '''MU = majority-utility winner
M = majority winner is not utility winner
CU = condorcet-utility winner
CPU = condorcet-plurality-utility winner
C = Condorcet winner is not utility winner
CP = condorcet-plurality winner is not utility winner
nc = No condorcet winner.
'''
    
    # place a text box in upper left in axes coords
    props = dict(facecolor='white', alpha=0.5)
    ax.text(0.4, 0.9, string, transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=props)
        
    plt.suptitle('%s-Dimensional Election, %d simulations' % (key, sim_num))
    # plt.savefig('scenario-categories-%s.png' % key)
    



    ###############################################################################
    plots.subplot_2row()
    plt.subplot(2, 1, 1)
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    ax, dfp = plots.heatmap(x='plurality_ratio', y='args.etype', hue='regret',
                  data=df, xbin=bins, vmax=25)
    plt.xlabel('% plurality winner ratio')
    plt.ylabel('')
    plt.title('% Average Voter Regret vs Plurality Ratio')
    
    ###############################################################################
    
    df = post.categorize_condorcet(df)
    ysortkey = dfp.index.values
    xsortkey = counts.index.values    
    
    plt.subplot(2, 1, 2)
    ax, dfp = plots.heatmap(x='categories', y='args.etype', hue='regret',
                  data=df,
                  xsortkey=xsortkey,
                  ysortkey=ysortkey,
                  vmax=50)
    plt.ylabel('')
    #ax.set_yticklabels('')
    plt.title('% Average Voter Regret vs Category')
    plt.xlabel('Scenario Categories')
    plt.subplots_adjust(left=.185, wspace=.025)    
    
    plt.suptitle('%s-Dimensional Election, %d simulations' % (key, sim_num))
    plt.savefig('regrets-vse-%d.png' % key)
    

    
    
    ###############################################################################


# %% Check no condorcet
isdim1 = df['args.user.num_dimensions'] == 1
notC = df['output.winner_categories.is_condorcet'] == -1
condition = notC & isdim1
print(np.sum(condition))


