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
from votesim.benchmarks import plots
dirname = votesim.definitions.DIR_DATA_BENCHMARKS
dirname = os.path.join(dirname, SimpleThreeWay.name)


@globalcache.cache_decorate('read')
def read():
    return SimpleThreeWay.read(dirname=dirname)


g = globalcache.create(globals())
p = read()
df = p.dataframe

keys = [
        
 'args.name',
 'type',
 'args.user.num_voters',
 'args.user.num_candidates',
 'args.user.num_dimensions',
 'args.user.strategy',
 'args.user.voter_tolerance',
 'output.voter.data_dependencies',
 'output.voter.pref_mean',
 'output.voter.pref_median',
 'output.voter.pref_std',
 'output.voter.regret_mean',
 'output.voter.regret_median',
 'output.voter.regret_random_avg',
 'output.candidate.data_dependencies',
 'output.candidate.pref',
 'output.candidate.regret_avg',
 'output.candidate.regret_best',
 'output.candidate.regrets',
 'output.candidate.winner_condorcet',
 'output.candidate.winner_majority',
 'output.candidate.winner_plurality',
 'output.candidate.winner_utility',
 'output.winner.data_dependencies',
 'output.winner.regret',
 'output.winner.regret_efficiency_candidate',
 'output.winner.regret_normed',
 'output.winner.regret_efficiency_voter',
 'output.winner.ties',
 'output.winner.winners',
 'output.winner_categories.data_dependencies',
 'output.winner_categories.is_condorcet',
 'output.winner_categories.is_majority',
 'output.winner_categories.is_utility',
 'output.ballot.bullet_num',
 'output.ballot.bullet_ratio',
 'output.ballot.data_dependencies',
 'output.ballot.full_num',
 'output.ballot.full_ratio',
 'output.ballot.marked_avg',
 'output.ballot.marked_num',
 'output.ballot.marked_std',
 ]
    
    


yname = 'args.etype'
xname = 'output.winner.regret_efficiency_candidate'

no_majority = df['output.candidate.winner_majority'] == -1
no_condorcet = df['output.candidate.winner_condorcet'] == -1
regret = 100* (1 - df['output.winner.regret_efficiency_candidate'])
pratio = df['output.candidate.plurality_ratio'] * 100



df['plurality_ratio'] = pratio
df['no_majority'] = no_majority
df['no_condorcet'] = no_condorcet
df['regret'] = regret

plt.figure()
bins = [30, 40, 50, 60, 70, 80, 90, 100]
plots.heatmap(x='plurality_ratio', y='args.etype', hue='regret',
              data=df, xbin=bins, vmax=25)


plt.figure()
plots.heatmap(x='plurality_ratio', y='args.etype', hue='regret',
              data=df, xbin=bins, func='count')







