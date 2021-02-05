# -*- coding: utf-8 -*-

import os
import itertools

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import globalcache
import definitions

sns.set()

import votesim
from votesim.benchmarks import simple
from votesim import plots

benchmark = simple.simple3way()

dirname = definitions.DIR_DATA_BENCHMARKS
dirname = os.path.join(dirname, benchmark.name)


@globalcache.cache_decorate('read')
def read():
    return benchmark.read(dirname=dirname)



# %% Read 



g = globalcache.create(globals())
p = read()
df = p.dataframe

# %% Post 
## Retrieve elections where IRV and top_two results disagree

os.makedirs('images', exist_ok=True)
os.chdir('images')


yname = 'args.etype'
xname = 'output.winner.regret_efficiency_voter'
# xname = 'output.winner.regret_efficiency_candidate'

vseed = 'args.voter-0.0.init.seed'
cseed = 'args.candidate.0.set_seed.seed'

df1 = df.copy()

# Filter by voting methods and tolerance
method1 = 'score5'

i1 = df1[yname] == method1
i3 = df1['args.user.voter_tolerance'] == 3.0
inew = i1 & i3
df1 = df1.loc[inew]
# df1 = df1.loc[inew].reset_index(drop=False)
# df1 = df1.set_index(p.parameters)


# df2 = df1[[
#            'index', 
#            vseed,
#            cseed,
#            yname, 
#            xname,
#            ]]

regrets = 1 - df1[xname]

# %% Re-run

# ii = 0
# i_irv = df2.values[ii, 0]
# i_tt = df2.values[ii, 1]
# e1 = benchmark.rerun(index=i_irv, df=df)
# e2 = benchmark.rerun(index=i_tt,  df=df)

ii = np.argmax(regrets)
index = regrets.index.values[ii]
max_regret = regrets.values[ii]

e1 = benchmark.rerun(index=index, df=df1)

output = e1.result.output

v_pref = e1.voters.data.pref
c_pref = e1.candidates.data.pref
bins = np.arange(-3, 3.5, .25)

plt.figure()
plt.subplot(2,1,1)
plots.vset()
plt.title('Score Utility Failure Example')
ax = sns.distplot(v_pref, kde=False, bins=bins, 
                  norm_hist=True, label='voter pref.')
plt.axvline(c_pref[0], ls='--', color='blue', alpha=.25,
            label='Candidate #0')
plt.axvline(c_pref[1], ls='--', color='orange', alpha=.25,
            label='Candidate #1')
plt.axvline(c_pref[2], ls='--', color='green', alpha=.45,
            label='Candidate #2, Winner')
median = output['output.voters.pref_median']
mean = output['output.voters.pref_mean']
plt.axvline(median, ls='-', color='black', alpha=.25,
            label='voter median pref.')
plt.axvline(mean, ls='--', color='black', alpha=.25, 
            label='voter mean pref.')
# xlim = ax.get_xlim()

counts = e1.result.runner.output['tally']
counts = counts / np.sum(counts)
plt.bar(c_pref.ravel(), counts, width=.2, alpha=.25, color='red', 
        label='Election Scores')
plt.xlim(-3, 4.)
plt.legend()
plt.xlabel('Preferences')


###

plt.subplot(2, 1, 2)
ii_sort = np.argsort(v_pref.ravel())
# scores = e1.used_ballots.scores
scores = np.round(e1.result.ballots * 5) 
v_scores1 = scores[ii_sort, 0] 
v_scores2 = scores[ii_sort, 1] + .1
v_scores3 = scores[ii_sort, 2] + .2
plt.plot(v_pref[ii_sort], v_scores1, '.-', alpha=.45, label='Candidate #0')
plt.plot(v_pref[ii_sort], v_scores2, '.-', alpha=.45, label='Candidate #1')
plt.plot(v_pref[ii_sort], v_scores3, '.-', alpha=.45, label='Candidate #2')
plt.xlim(-3, 4.)
plt.ylabel('Scores given by voter')
plt.xlabel('Preferences')
plt.axvline(*c_pref[0], ls='--', color='blue', alpha=.25,)
plt.axvline(c_pref[1], ls='--', color='orange', alpha=.25,)
plt.axvline(c_pref[2], ls='--', color='green', alpha=.45,)
plt.legend()
plt.savefig('score_study.png')

# place a text box in upper left in axes coords
# string = """Existence of Candidate #1 causes Candidate #0 voters to rate 
# Candidate #2 higher, as #2 is relatively prefered compared to #1. Voter 
# behavior and the flexibility of the score scale causes score voting 
# to fail Independence of Irrelevant Alternatives (IIA)."""
# props = dict(facecolor='white', alpha=0.5)
# ax.text(0.4, 0.9, string, transform=ax.transAxes, fontsize=10,
#         verticalalignment='top',
#         horizontalalignment='left',
#         bbox=props)

# %% Re-run with only two candidates

v = e1.voters
c = e1.candidates
cpref = np.delete(c.data.pref, 1, axis=0)
c.candidates = cpref
e1.set_models(v, c)
e1.run('score5')
print(e1.result.winners)


