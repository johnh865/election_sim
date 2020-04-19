# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from itertools import cycle


import votesim
from votesim.models import spatial
from votesim.utilities.write import StringTable

import seaborn as sns
import matplotlib.pyplot as plt



###############################################################################

e = spatial.load_election('election3way.pkl')
metric_name = 'stats.regret.efficiency_voter'
df0 = e.dataframe()

candidates = df0['args.candidate.0.add.candidates']
candidates = np.column_stack(candidates)
candidates = np.sort(candidates, axis=1)

df0['candidate-1'] = candidates[0]
df0['candidate-2'] = candidates[1]
df0['candidate-3'] = candidates[2]
winners = np.concatenate(df0['stats.winner.all'])
best = np.concatenate(df0['stats.candidate.best'])
trials = np.arange(len(best))

winlocs = candidates[winners, trials]
bestlocs = candidates[best, trials]
df0['winner-locs'] = winlocs
df0['best-locs'] = bestlocs





df1 = df0.sort_values(by=metric_name)



###############################################################################

groupby = df1.groupby('args.election.2.run.etype')
names = list(groupby.groups.keys())

vse_table = []
for name in names:
    dfi = groupby.get_group(name)
    vsei = dfi[metric_name]
    vse_table.append(vsei)
vse_table = np.array(vse_table, dtype=float)

###############################################################################
### Construct histogram
voterhist, hedges = np.histogram(e.voters.voters, bins=20, density=True)
ch_edges = .5 * (hedges[0:-1] + hedges[1:])





#sns.lineplot()






