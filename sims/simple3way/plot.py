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
from votesim import plots
import definitions

benchmark = simple.simple3way()

dirname = definitions.DIR_DATA_BENCHMARKS
dirname = os.path.join(dirname, benchmark.name)


@globalcache.cache_decorate('read')
def read():
    return benchmark.read(dirname=dirname)


def categorize(df):
    """
    Category Combinations
    
    Labels 
    -------
    - M = majority winner
    - P = plurality winner
    - C = condorcet winner 
    - U = utility winner
    
    Categories
    ----------
    - MU = Has majority utility winner
    - M = Has majority winner that is not utility winner.
    -
    - CPU = Has condorcet, utility, plurality winner
    - CU = Has condorcet, utility winner that is not plurality winner
    - CP = Has condorcet, plurality winner that is not utility winner
    - C = Has condorcet winner who is not plurality and utility winner
    - 
    - NC = Has no Condorcet winner
    - 
    """
    
    iM = df['output.candidates.winner_majority']
    iP = df['output.candidates.winner_plurality']
    iC = df['output.candidates.winner_condorcet']
    iU = df['output.candidates.winner_utility']
    
    df = df.copy()
    df.loc[:, 'categories'] = 'No category'
    
    maj = iM > -1
    no_maj = ~maj
    
    MU = (iM == iU)
    M = maj & (iM != iU)
    
    CPU = no_maj & (iC == iP) & (iC == iU)
    CP  = no_maj & (iC == iP) & (iC != iU)
    CU  = no_maj & (iC == iU) & (iC != iP)
    C = (iC > -1) & (iC != iP) & (iC != iU)
    
    PU  = no_maj & (iP == iU) & (iP != iC)
    NC = (iC == -1)
    
    df.loc[MU, 'categories'] = 'MU'
    df.loc[M, 'categories'] = 'M'
    df.loc[CPU, 'categories'] = 'CPU'
    df.loc[CP, 'categories'] = 'CP'
    df.loc[CU, 'categories'] = 'CU'
    df.loc[C, 'categories'] = 'C'
    df.loc[PU, 'categories'] = 'PU'
    df.loc[NC, 'categories'] = 'nc'
    
    return df
    
    
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
xname = 'output.winner.regret_efficiency_voter'

no_majority = df['output.candidates.winner_majority'] == -1
no_condorcet = df['output.candidates.winner_condorcet'] == -1
regret = 100* (1 - df[xname])
pratio = df['output.candidates.plurality_ratio'] * 100

df = df.reset_index()
df.loc[:, 'plurality_ratio'] = pratio
df.loc[:, 'no_majority'] = no_majority
df.loc[:, 'no_condorcet'] = no_condorcet
df.loc[:, 'regret'] = regret


### Categorize scenario parameters
arg_tol = df['args.user.voter_tolerance']
groupby = df.groupby(by='args.user.voter_tolerance')
keys = groupby.groups.keys()
dframes = (groupby.get_group(k) for k in keys)




# %% Plot categories
### Plot election categories
df = groupby.get_group(list(keys)[0])
etype_num = len(df['args.etype'].unique())
sim_num = len(df) / etype_num


plots.vset()
plots.subplot_2row()
plt.subplot(2, 1, 1)
# sns.distplot(a=pratio, bins=10, norm_hist=True, kde=False)
sns.histplot(data=np.array(pratio), stat='probability', bins=10,kde=False)
plt.xlabel('% plurality winner ratio')
plt.ylabel('Scenario probability')
plt.title('Probability of Plurality Ratio in Benchmark')

plt.subplot(2, 1, 2)
df = categorize(df)
c = df['categories']
counts = c.value_counts() / len(c)*100
# sns.barplot(x=counts.keys(), y=counts.values)
ax = plots.bar(x=counts.keys(), y=counts.values, fmt='g')
plt.ylabel('% Occurrence')
# sns.countplot(x='categories', data=df,)
plt.xlabel('Scenario Categories')
plt.title('Majority/Condorcet/Utility/Plurality Occurrences')


string = '''MU = majority-utility winner
CU = condorcet-utility winner
CPU = condorcet-plurality-utility winner
M = majority winner is not utility winner
PU = plurality-utility winner
nc = No condorcet winner.
CP = condorcet-plurality winner is not utility winner'''

# place a text box in upper left in axes coords
props = dict(facecolor='white', alpha=0.5)
ax.text(0.4, 0.9, string, transform=ax.transAxes, fontsize=10,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=props)
    
plt.suptitle('3-Way Election, 1-Dimensional, %d simulations' % sim_num)
plt.savefig('scenario-categories.png')


# %% Plot heatmaps
i = 0
for key, df in zip(keys, dframes):
    
    # plt.figure(figsize=(12,8))
    
    plots.subplot_2row()
    plt.subplot(2, 1, 1)
    bins = [0, 30, 40, 50, 60, 70, 80, 90, 100]
    ax, dfp = plots.heatmap(x='plurality_ratio', y='args.etype', hue='regret',
                  data=df, xbin=bins, vmax=25)
    plt.xlabel('% plurality winner ratio')
    plt.ylabel('')
    plt.title('% Average Voter Regret vs Plurality Ratio')
    
    # plt.hist(pratio, density=True, )
    # hist, _ = np.histogram(pratio, bins=bins,) / len(pratio)
    # plots.bar(x=bins, )
    


    ###############################################################################
    
    df = categorize(df)
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
    
    plt.suptitle('3-Way Election, 1-Dimensional, voter tolerance=%s, '
                 '%d simulations' % (key, sim_num))
    plt.savefig('regrets-%d.png' % i)
    
    i += 1

    
    
    
    ###############################################################################





