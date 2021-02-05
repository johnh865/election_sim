# -*- coding: utf-8 -*-
"""

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

"""
import dataclasses
import pdb
import numpy as np
import votesim
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


from votesim.models.spatial import Candidates, Voters, Election

from votesim.benchmarks.tactical_v2 import tactical_model_v2

votesim.plots.vset()

e = tactical_model_v2(
    name='test', 
    methods=['plurality'], 
    ndim=2,
    cnum=5,
    seed=16,
    numvoters=201)


df = e.dataframe()
df1 = df[[
    'args.user.eid',
    'args.user.strategy',
    'args.name',
    'args.etype',
    'args.strategy.0.add.subset',
    'args.strategy.0.add.underdog',
    'args.strategy.0.add.tactics',
    'output.winner.regret_efficiency_candidate',
    ]]
# pdb.set_trace()


class Result(object):
    def __init__(self, iloc: int):
        e1 = e.rerun(df.iloc[iloc])
        cdata = e1.candidates.data
        vdata = e1.voters.data
        voter_pref = vdata.pref
        cand_pref = cdata.pref
        result = e1.result
        ballots = result.ballots 
        vlocs = np.where(ballots == 1)
        cnum = cdata.pref.shape[0]
            
        # for ii in range(cnum):
        #     supporters = voter_pref[vlocs[1] == ii]
            # x = supporters[:, 0]
            # y = supporters[:, 1]    
            
        self.cdata = cdata
        self.vdata = vdata
        self.voter_pref = voter_pref
        self.cand_pref = cand_pref
        self.result = result
        self.ballots = ballots
        # self.supporters = supporters
        self.cnum = cnum
        self.underdog = df.iloc[iloc]['args.strategy.0.add.underdog']
        return 



# %% Plot honest

candidate_color_map = [
    'Red', 'Blue', 'Green', 'Yellow', 'Cyan']
# plot candiates


# plot voters

colors = ['r','b','g','y','c']*5
plt.figure(figsize=(6.5,6.5))
# HONEST
result = Result(0)
# plt.subplot(4,2,1)
for ii in range(result.cnum):
    vlocs = np.where(result.ballots == 1)
    supporters = result.voter_pref[vlocs[1] == ii]    
    x = supporters[:, 0]
    y = supporters[:, 1]
    plt.scatter(x, y, c=colors[ii], alpha=.5)
    
    plt.plot(result.cand_pref[ii,0], result.cand_pref[ii, 1],
             '*', 
             markersize=15,
             markeredgecolor='k',
             color=colors[ii])
    plt.text(result.cand_pref[ii,0], result.cand_pref[ii, 1],
             '   %d# of voters' % len(x),
             ha='left',
             va='center',
             fontsize='small',
             color='k')
    plt.title('Honest Voters')
plt.xlabel("Preference #1")
plt.ylabel("Preference #2")
plt.tight_layout()

legend1= mlines.Line2D([], [],
                       marker='*',
                       linewidth=0,
                       color='gray', 
                       markersize=15,
                       markeredgecolor='k',
                       )
legend2= mlines.Line2D([], [],
                       marker='o',
                       linewidth=0,
                       color='gray', 
                       alpha=0.5,
                       )

plt.legend([legend1, legend2], ['candidate', 'voter'])
plt.savefig('sample-fptp-honest.png')

honest_winner = candidate_color_map[result.result.winners[0]]

# %%  One Side Strategy #1
# One Side Strategy #1
plt.figure(figsize=(6.5,7.0))

result = Result(1)
plt.subplot(2,2,1)
for ii in range(result.cnum):
    vlocs = np.where(result.ballots == 1)
    supporters = result.voter_pref[vlocs[1] == ii]    
    x = supporters[:, 0]
    y = supporters[:, 1]
    plt.scatter(x, y, c=colors[ii], alpha=.5)
    
    plt.plot(result.cand_pref[ii,0], result.cand_pref[ii, 1],
             '*', 
             markersize=15,
             markeredgecolor='k',
             color=colors[ii])
    plt.text(result.cand_pref[ii,0], result.cand_pref[ii, 1],
             '   %d# of voters' % len(x),
             ha='left',
             va='center',
             fontsize='small',
             color='k')
    
underdog_name = candidate_color_map[result.underdog]
plt.title(f'One Sided, {underdog_name} Coalition')


# Two Side Strategy #1
result = Result(2)
plt.subplot(2,2,2)
for ii in range(result.cnum):
    vlocs = np.where(result.ballots == 1)
    supporters = result.voter_pref[vlocs[1] == ii]    
    x = supporters[:, 0]
    y = supporters[:, 1]
    plt.scatter(x, y, c=colors[ii], alpha=.5)
    
    plt.plot(result.cand_pref[ii,0], result.cand_pref[ii, 1],
             '*', 
             markersize=15,
             markeredgecolor='k',
             color=colors[ii])
    plt.text(result.cand_pref[ii,0], result.cand_pref[ii, 1],
             '   %d# of voters' % len(x),
             ha='left',
             va='center',
             fontsize='small',
             color='k')
    
plt.title(f'Two Sided, {underdog_name} vs {honest_winner}')




# One Side Strategy #2
result = Result(3)
plt.subplot(2,2,3)
for ii in range(result.cnum):
    vlocs = np.where(result.ballots == 1)
    supporters = result.voter_pref[vlocs[1] == ii]    
    x = supporters[:, 0]
    y = supporters[:, 1]
    plt.scatter(x, y, c=colors[ii], alpha=.5)
    
    plt.plot(result.cand_pref[ii,0], result.cand_pref[ii, 1],
             '*', 
             markersize=15,
             markeredgecolor='k',
             color=colors[ii])
    plt.text(result.cand_pref[ii,0], result.cand_pref[ii, 1],
             '   %d# of voters' % len(x),
             ha='left',
             va='center',
             fontsize='small',
             color='k')
underdog_name = candidate_color_map[result.underdog]
plt.title(f'One Sided, {underdog_name} Coalition')

# Two Side Strategy #2
result = Result(4)
plt.subplot(2,2,4)
for ii in range(result.cnum):
    vlocs = np.where(result.ballots == 1)
    supporters = result.voter_pref[vlocs[1] == ii]    
    x = supporters[:, 0]
    y = supporters[:, 1]
    plt.scatter(x, y, c=colors[ii], alpha=.5)
    
    plt.plot(result.cand_pref[ii,0], result.cand_pref[ii, 1],
             '*', 
             markersize=15,
             markeredgecolor='k',
             color=colors[ii])
    plt.text(result.cand_pref[ii,0], result.cand_pref[ii, 1],
             '   %d# of voters' % len(x),
             ha='left',
             va='center',
             fontsize='small',
             color='k')
plt.title(f'Two Sided, {underdog_name} vs {honest_winner}')
    
plt.tight_layout()
plt.savefig('sample-fptp-tactical-1.png')


# %% Plot strategy 

plt.figure(figsize=(6.5,7.0))

# One Side Strategy #3
result = Result(5)
plt.subplot(2,2,1)
for ii in range(result.cnum):
    vlocs = np.where(result.ballots == 1)
    supporters = result.voter_pref[vlocs[1] == ii]    
    x = supporters[:, 0]
    y = supporters[:, 1]
    plt.scatter(x, y, c=colors[ii], alpha=.5)
    
    plt.plot(result.cand_pref[ii,0], result.cand_pref[ii, 1],
             '*', 
             markersize=15,
             markeredgecolor='k',
             color=colors[ii])
    plt.text(result.cand_pref[ii,0], result.cand_pref[ii, 1],
             '   %d# of voters' % len(x),
             ha='left',
             va='center',
             fontsize='small',
             color='k')
    
underdog_name = candidate_color_map[result.underdog]
plt.title(f'One Sided, {underdog_name} Coalition')

# Two Side Strategy #3
result = Result(6)
plt.subplot(2,2,2)
for ii in range(result.cnum):
    vlocs = np.where(result.ballots == 1)
    supporters = result.voter_pref[vlocs[1] == ii]    
    x = supporters[:, 0]
    y = supporters[:, 1]
    plt.scatter(x, y, c=colors[ii], alpha=.5)
    
    plt.plot(result.cand_pref[ii,0], result.cand_pref[ii, 1],
             '*', 
             markersize=15,
             markeredgecolor='k',
             color=colors[ii])
    plt.text(result.cand_pref[ii,0], result.cand_pref[ii, 1],
             '   %d# of voters' % len(x),
             ha='left',
             va='center',
             fontsize='small',
             color='k')
plt.title(f'Two Sided, {underdog_name} vs {honest_winner}')
    
    
# One Side Strategy #4
result = Result(7)
plt.subplot(2,2,3)
for ii in range(result.cnum):
    vlocs = np.where(result.ballots == 1)
    supporters = result.voter_pref[vlocs[1] == ii]    
    x = supporters[:, 0]
    y = supporters[:, 1]
    plt.scatter(x, y, c=colors[ii], alpha=.5)
    
    plt.plot(result.cand_pref[ii,0], result.cand_pref[ii, 1],
             '*', 
             markersize=15,
             markeredgecolor='k',
             color=colors[ii])
    plt.text(result.cand_pref[ii,0], result.cand_pref[ii, 1],
             '   %d# of voters' % len(x),
             ha='left',
             va='center',
             fontsize='small',
             color='k')
    
underdog_name = candidate_color_map[result.underdog]
plt.title(f'One Sided, {underdog_name} Coalition')

# Two Side Strategy #3
result = Result(8)
plt.subplot(2,2,4)
for ii in range(result.cnum):
    vlocs = np.where(result.ballots == 1)
    supporters = result.voter_pref[vlocs[1] == ii]    
    x = supporters[:, 0]
    y = supporters[:, 1]
    plt.scatter(x, y, c=colors[ii], alpha=.5)
    
    plt.plot(result.cand_pref[ii,0], result.cand_pref[ii, 1],
             '*', 
             markersize=15,
             markeredgecolor='k',
             color=colors[ii])
    plt.text(result.cand_pref[ii,0], result.cand_pref[ii, 1],
             '   %d# of voters' % len(x),
             ha='left',
             va='center',
             fontsize='small',
             color='k')
plt.title(f'Two Sided, {underdog_name} vs {honest_winner}')
    
plt.tight_layout()
plt.savefig('sample-fptp-tactical-2.png')
