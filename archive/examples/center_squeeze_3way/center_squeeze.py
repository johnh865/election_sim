# -*- coding: utf-8 -*-

"""
Honest election simulations comparing IRV, plurality, and score voting

"""
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

types = ['irv', 'score', 'plurality', 'smith_minimax']



v = spatial.SimpleVoters(0)
v.add_random(500, ndim=1)
c = spatial.Candidates(v, 0)
e = spatial.Election(v, None, seed=0)

distances = np.linspace(0, 2, 50)
trials = np.arange(50)
for dist in distances:
    c.reset()
    
    a = [-dist, 0, dist+.05]
    a = np.atleast_2d(a).T
    c.add(a)
    e.set_data(v, c)
    
    for t in types:
        e.run(etype=t)
        

df = e.dataframe()
vse = df['stats.regret.efficiency_voter']
vse1 = vse[df['args.election.1.run.etype'] == 'irv']
vse2 = vse[df['args.election.1.run.etype'] == 'score']
vse3 = vse[df['args.election.1.run.etype'] == 'plurality']
vse4 = vse[df['args.election.1.run.etype'] == 'smith_minimax']



candidates1 = df['args.candidate.0.add.candidates']
candidates1 = candidates1[df['args.election.1.run.etype'] == 'irv']
candidates1 = np.column_stack(candidates1)


plt.figure()
plt.suptitle("Center Sqeeze Suceptibility, Symmetric 3 Candidate Race")
plt.subplot(2,2,1)
plt.hist(v.voters, bins=20)
plt.xlabel('Voter Preference Location (std deviations)')
plt.ylabel('Voter Distribution')
plt.grid()
 
plt.subplot(2,2,2)
plt.plot(distances, vse1, '.-', label='irv')
plt.plot(distances, vse2, '.-', label='score')
plt.plot(distances, vse3, '--', label='plurality')
plt.plot(distances, vse4, '--', label='smith_minimax')
plt.legend()
plt.xlabel('Edge candidate distance from voter centroid (std deviations)')
plt.ylabel('voter satisfaction efficiency')
plt.grid()

plt.subplot(2,2,3)
plt.plot(trials, candidates1[0],'.', label='left candidate')
plt.plot(trials, candidates1[1],'.', label='center candidate')
plt.plot(trials, candidates1[2],'.', label='right candidate')
plt.xlabel("Simulation Trial Number")
plt.ylabel("Candidate preference location (std deviations)")
plt.legend()
plt.grid()