# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 18:22:10 2020

@author: John

Interesting Metrics to Pursue
------------------------------
- # of viable underdogs 
- Worst one-sided regret for given voter/candidate/method combo that is effective for underdog
- Worst two-sided regret for given voter/candidate/method combo that is effective

Output keys
-----------
args.user.eid
args.user.strategy
args.name
args.etype
args.candidate.0.set_seed.seed
args.candidate.1.add_random.cnum
args.candidate.1.add_random.sdev
args.voter-0.0.init.seed
args.voter-0.0.init.order
args.voter-0.1.set_behavior.tol
args.voter-0.1.set_behavior.base
args.voter-0.2.add_random.numvoters
args.voter-0.2.add_random.ndim
args.election.0.init.seed
args.election.0.init.numwinners
args.election.0.init.scoremax
args.election.0.init.name
args.election.1.run.etype
output.voters.pref_mean
output.voters.pref_median
output.voters.pref_std
output.voters.regret_mean
output.voters.regret_median
output.voters.regret_random_avg
output.candidates.plurality_ratio
output.candidates.pref
output.candidates.regret_avg
output.candidates.regret_best
output.candidates.regrets
output.candidates.winner_condorcet
output.candidates.winner_majority
output.candidates.winner_plurality
output.candidates.winner_utility
output.winner.regret
output.winner.regret_efficiency_candidate
output.winner.regret_efficiency_voter
output.winner.regret_normed
output.winner.ties
output.winner.winners
output.winner_categories.is_condorcet
output.winner_categories.is_majority
output.winner_categories.is_utility
output.ballot.bullet_num
output.ballot.bullet_ratio
output.ballot.full_num
output.ballot.full_ratio
output.ballot.marked_avg
output.ballot.marked_num
output.ballot.marked_std
args.strategy.0.add.ratio
args.strategy.0.add.subset
args.strategy.0.add.underdog
args.strategy.0.add.tactics
output.tactic_compare.regret.topdog-0
output.tactic_compare.regret.underdog-0
output.tactic_compare.regret.tactical-0
output.tactic_compare.regret.honest-0
output.tactic_compare.regret_efficiency_candidate.topdog-0
output.tactic_compare.regret_efficiency_candidate.underdog-0
output.tactic_compare.regret_efficiency_candidate.tactical-0
output.tactic_compare.regret_efficiency_candidate.honest-0
output.tactic_compare.regret_efficiency_voter.topdog-0
output.tactic_compare.regret_efficiency_voter.underdog-0
output.tactic_compare.regret_efficiency_voter.tactical-0
output.tactic_compare.regret_efficiency_voter.honest-0
output.tactic_compare.voter_nums.topdog-0
output.tactic_compare.voter_nums.underdog-0
output.tactic_compare.voter_nums.tactical-0
output.tactic_compare.voter_nums.honest-0
args.strategy.1.add.ratio
args.strategy.1.add.subset
args.strategy.1.add.tactics
args.strategy.1.add.underdog

Analysis Assumptions
--------------------
- Rational voters; strategy only used if success is guaranteed. 
- For two-sided, only top-dog bullet voting is considered.

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
from votesim.benchmarks import tactical_v2
from votesim import plots

benchmark = tactical_v2.tactical_v2_1()

dirname = votesim.definitions.DIR_DATA_BENCHMARKS
dirname = os.path.join(dirname, benchmark.name)


# %% Initialize global cache
@globalcache.cache_decorate('read')
def read():
    return benchmark.read(dirname=dirname)

g = globalcache.create(globals())
p = read()
df = p.dataframe


# %% Define aggregation functions
p95 = lambda x : np.percentile(x, 95)
p05 = lambda x : np.percentile(x, 5)
p95.__name__ = 'percentile90'
p05.__name__ = 'percentile10'

# %% Get honest results
index = df['args.user.strategy'] == 'honest'
df0 = df[index][['args.etype', 'output.winner.regret_efficiency_candidate']]
groupby = df0.groupby(by='args.etype')
honest_vse_mean = groupby.agg('mean')
honest_vse_p95 = groupby.agg(p95)
honest_vse_p05 = groupby.agg(p05)



# %% Get 1-sided strategy results
print("Get 1-sided data")
index1 = df['args.user.strategy'] == 'one-sided' 
index2 = index1 | index
columns = [
    'args.etype', 
    'args.user.eid',
    'args.user.strategy',
    'args.strategy.0.add.underdog',
    'args.strategy.0.add.tactics',
    'output.winner.regret_efficiency_candidate',
    'output.tactic_compare.regret_efficiency_candidate.underdog-0'
  ]
df1 = df[columns].loc[index2]



def myfilter(dframe):
    """Only return best performing underdog election for underdog."""
    vse = dframe['output.tactic_compare.regret_efficiency_candidate.underdog-0']
    vse_under_best = np.max(vse)
    new = dframe[vse == vse_under_best]
    return new.iloc[0]
    

def myfilter2(dframe):
    """Return best performing underdog election. If all strategy 
    backfires, return honest result."""
    

    # Don't need to get these, vse1 >= 0 will filter out honest
    # Get 1-sided result locations
    # iloc_onesided = df['args.user.strategy'] == 'one-sided'
    
    # Ignore strategies that backfire. 
    vse1 = dframe['output.tactic_compare.regret_efficiency_candidate.underdog-0']
    dframe1 = dframe.loc[vse1 >= 0]

    # Retrieve most effective strategy with worst VSE.
    vse = dframe1['output.winner.regret_efficiency_candidate']
    vse_worst = np.min(vse)
    new = dframe1.loc[vse == vse_worst]
    if len(new) > 0:
        return new.iloc[0]
    else:
        # Get honest result location
        iloc_honest = dframe['args.user.strategy'] == 'honest'        
        return dframe.loc[iloc_honest].iloc[0]
    
    

# Filter elections for best possible underdog strategy result. 
print("Group 1-sided data")
groupby = df1.groupby(by=['args.etype', 'args.user.eid'])
print("Filter for the best performing tactics")
df1p2 = groupby.apply(myfilter2).reset_index(drop=True)
df1p3 = df1p2[['args.etype', 'output.winner.regret_efficiency_candidate']]
groupby = df1p3.groupby(by='args.etype')
onesided_vse_mean = groupby.agg('mean')
onesided_vse_p95 = groupby.agg(p95)
onesided_vse_p05 = groupby.agg(p05)


# %% Get 2-sided strategy results


def myfilter_2sided(dframe):
    """Return worst performing election, only when underdog strategy is effective."""
    vse = dframe['output.winner.regret_efficiency_candidate']    
    vse_worst = np.min(vse)
    new = dframe[vse == vse_worst]
    return new.iloc[0]
    

index= df['args.user.strategy'] == 'two-sided'
df2 = df[index][[
    'args.etype',
    'args.user.eid',
    'output.winner.regret_efficiency_candidate',
    'args.strategy.0.add.underdog',
    'args.strategy.0.add.tactics',
    'args.strategy.1.add.tactics',
    'output.tactic_compare.regret_efficiency_candidate.underdog-0',
    'output.tactic_compare.regret_efficiency_candidate.topdog-0',
    'output.tactic_compare.voter_nums.topdog-0',
    'output.tactic_compare.voter_nums.underdog-0',    
    
    ]]
df222 = df2[df2['args.etype'] == 'ranked_pairs']
# df222 = df2[df2['args.etype'] == 'approval50']
# Get corresponding 1-sided results for the 2-sided election. 



groupby = df2.groupby(by=['args.etype', 'args.user.eid'])
df2p2 = groupby.apply(myfilter_2sided).reset_index(drop=True)
df2p3 = df2p2[['args.etype', 'output.winner.regret_efficiency_candidate']]
groupby = df2p3.groupby(by='args.etype')

twosided_vse_mean = groupby.agg('mean')
twosided_vse_p95 = groupby.agg(p95)
twosided_vse_p05 = groupby.agg(p05)



net_scores = (onesided_vse_mean + twosided_vse_mean + honest_vse_mean) / 3
net_scores = net_scores.sort_values(by='output.winner.regret_efficiency_candidate')


# %% PLOT


fig, ax = plt.subplots(figsize=[6.5, 7])
# plots.plot_1set()
methods = net_scores.index
colors = plt.get_cmap()
votesim.plots.vset()

for ii, method in enumerate(methods):
    if ii == 0:
        lstart = ''
    else:
        lstart = '_'
    
    honest_p05 = honest_vse_p05.loc[method]
    honest_mean = honest_vse_mean.loc[method]
    honest_p95 = honest_vse_p95.loc[method]
    width = honest_p95 - honest_p05
    plt.barh(y=ii + .3, 
             left=honest_p05, 
             height=0.3,
             width=width,
             color='blue',
             edgecolor='k',
             alpha=.15,
             label=lstart+'honest')

    twosided_p05 = twosided_vse_p05.loc[method]
    twosided_mean = twosided_vse_mean.loc[method]
    twosided_p95 = twosided_vse_p95.loc[method]
    width = twosided_p95 - twosided_p05
    plt.barh(y=ii+ 0,
             left=twosided_p05,
             height=0.3,
             width=width,
             color='green',
             edgecolor='k',
             alpha=.15,
             label=lstart+'two-sided')  

    onesided_p05 = onesided_vse_p05.loc[method]
    onesided_mean = onesided_vse_mean.loc[method]
    onesided_p95 = onesided_vse_p95.loc[method]
    width = onesided_p95 - onesided_p05
    plt.barh(y=ii - .3,
             left=onesided_p05,
             height=0.3,
             width=width,
             color='red',
             edgecolor='k',
             alpha=.15,
             label=lstart+'one-sided')
    
    
    
    plt.text(honest_mean+.02, ii + .3,
             "%.2f" % honest_mean, 
             ha='left', va='center',
             color='blue', alpha=.9,
             size='small'
             )
    
    plt.text(twosided_mean+.02, ii - .0,
             "%.2f" % twosided_mean, 
             ha='left', va='center',
             color='green', alpha=.9,
             size='small'
             )
    
    plt.text(onesided_mean+.02, ii - .3,
             "%.2f" % onesided_mean, 
             ha='left', va='center',
             color='red', alpha=.9,
             size='small'
             )    


# plot mean
yticks = np.arange(len(methods))

plt.plot(honest_vse_mean.loc[methods], 
         yticks + 0.3, '|',
         markersize=10,
         color='blue', alpha=.9)
plt.plot(twosided_vse_mean.loc[methods],
         yticks, '|',
         markersize=10,
         color='green', alpha=.9)
plt.plot(onesided_vse_mean.loc[methods], 
         yticks - 0.3, '|',
         markersize=10,
         color='red', alpha=.9)



ax = plt.gca()
# ax.grid(False)
ax.set_yticks(yticks)
ax.set_yticklabels(methods)
plt.legend()

for tick in ax.yaxis.get_minor_ticks():
    tick.label1.set_verticalalignment('top')
    
plt.xlabel('Voter Satisfaction Efficiency')
plt.title('VSE at 5th percentile, mean, and 95th percentile')
plt.tight_layout()
plt.savefig('Tactical-VSE.png')
# plt.xlim(None, 1.05)
    
# %% Test violin

# df0['strategy'] = 'honest'
# df1p3['strategy'] = 'one-sided'
# df2['strategy'] = 'two-sided'
# df3 = pd.merge((df0, df1p3, df2))
# sns.boxplot(data=df0, 
#                x='args.etype',
#                y='output.winner.regret_efficiency_candidate',
#                fliersize=0)

    