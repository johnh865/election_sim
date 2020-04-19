# -*- coding: utf-8 -*-

"""
Honest election simulations comparing IRV, plurality, and score voting

"""
import itertools

import votesim
from votesim.models import spatial
from votesim.utilities.write import StringTable

import matplotlib.pyplot as plt
#import seaborn as sns

import numpy as np
import pandas as pd


#votesim.logconfig.setInfo()
#votesim.logconfig.setDebug()
votesim.logconfig.setWarning()

types = ['irv', 'score', 'star', 'plurality', 'smith_minimax']
metric_name = 'stats.regret.vsp'


v = spatial.SimpleVoters(0)
v.add_random(500, ndim=1)
c = spatial.Candidates(v, 0)
e = spatial.Election(v, None, seed=0, scoremax=5)

### Develop parametric study variables
distances = np.linspace(0, 2, 10)[1:]
skew = np.linspace(0,1, 10)[1:-1]
offset = np.linspace(0, 0.5, 10 )

### Loop through parameters
loop_product = itertools.product(distances, skew, offset)
for ii, (d, s, o) in enumerate(loop_product):
    print(ii, end=',')
    c.reset()
    a = np.array([0, s, 1]) - 0.5
    a = a * d + o
    a = np.atleast_2d(a).T
    c.add(a)
    e.set_data(v, c)
    
    args = {'distance':d, 'skew':s, 'offset':o}
    e.user_data(**args)
    
    for t in types:
        e.run(etype=t)

e.save_json('center_squeeze_part2_results.json')
e.save_csv('center_squeeze_part2_results.csv')

df = e.dataframe()
df1 = df.filter(items=('args.election.2.run.etype',
                       metric_name,
                      'args.election.1.user_data.distance',
                      'args.election.1.user_data.skew',
                      'args.election.1.user_data.offset',
                      'stats.winners',
                      ))


vse = df1[metric_name]

vse_table = [vse[df1['args.election.2.run.etype'] == t] for t in types]
vse_table = np.array(vse_table)

ii_vse_sorted = np.argsort(vse_table, axis=1)

typenum = len(types)
for jj in range(typenum):
    vse_table[jj] = vse_table[jj, ii_vse_sorted[jj]]


### Get voter histogram
voterhist, hedges = np.histogram(v.voters, bins=20, density=True)
ch_edges = .5 * (hedges[0:-1] + hedges[1:])

#
#
candidates1 = df['args.candidate.0.add.candidates']
candidates1 = candidates1[df['args.election.2.run.etype'] == 'irv']
candidates1 = np.column_stack(candidates1)


winners = df['stats.winners']


trials = np.arange(len(candidates1.T))
epercent = trials / np.max(trials) * 100


plt.figure()
plt.subplot(2,1,1)
plt.plot(trials, candidates1[0], '.', label='Left')
plt.plot(trials, candidates1[1], '.', label='Center')
plt.plot(trials, candidates1[2], '.', label='Right')
plt.axhline(0, color='k')

plt.ylabel('Candidate location')
plt.grid()
plt.legend()

xticks = np.arange(0, 101, 1)

plt.subplot(2,1,2)
for vsei, typei in zip(vse_table, types):
    plt.plot(epercent, vsei, label=typei)
plt.legend()
plt.grid()
plt.ylim(0, None)    
plt.ylabel('Voter Satisfaction**')
plt.xlabel('Percentile of Trials')



for jj, etype in enumerate(types):
    plt.figure()
    candidatesi = candidates1[:, ii_vse_sorted[jj]]
    winnersi = winners[df['args.election.2.run.etype'] == etype]
    winnersi = np.column_stack(winnersi).ravel()[ii_vse_sorted[jj]]
    winlocs = candidatesi[(winnersi, trials)]
    
    
    ax = plt.subplot(2,1,1)
    plt.plot(epercent, candidatesi[0], '.', label='Left')
    plt.plot(epercent, candidatesi[1], '.', label='Center')
    plt.plot(epercent, candidatesi[2], '.', label='Right')
    plt.plot(epercent, winlocs, 'o', alpha=.3, fillstyle='none', label='winner')
    
    
    
    plt.plot(voterhist*100, ch_edges, label='voters')
    
    plt.xlabel('Percentile of Trials')
    plt.ylabel('Candidate location')
    ax.set_xticks(xticks, minor=True)
    plt.xticks
    plt.axhline(0)
    plt.grid(b=True, which='both')
    plt.legend(ncol=3)
    
    ax = plt.subplot(2,1,2)
    ax.set_xticks(xticks, minor=True)
    
    plt.plot(epercent, vse_table[jj], label=etype)
    plt.xlabel('Percentile of Trials')
    plt.ylabel('Voter Satisfaction**')    
    plt.legend()
    plt.grid(b=True, which='both')
    plt.ylim(0, 1.1)    








