# -*- coding: utf-8 -*-
"""Test min/max strategy for scored voting.

Check both rated ballot "score5" and scored ballot "score"
"""
import numpy as np
import votesim
from votesim.models import spatial



voter_pref = [-1]*9 + [-.22]*5 + [0.5]*1 + [1]*5
voter_pref = np.array(voter_pref)[:, None]
candidate_pref = [-1, .5, 1]
candidate_pref = np.array(candidate_pref)[:, None]

etypes = ['score', 'score5']

for etype in etypes:
    
    v = spatial.Voters(seed=0)
    v.add(voter_pref)
    c = spatial.Candidates(v)
    c.add(candidate_pref)
    
    # run honest election
    print('---- HONEST ELECTION --------------------------------')
    
    e1 = spatial.Election(voters=v, candidates=c)
    result1 = e1.run(etype)
    tally = result1.runner.output['tally']
    print('ballots = ')
    print(result1.runner.ballots)
    print('honest tally = ', tally)
    print('candidate net regrets = ', result1.stats.candidates.regrets)
    assert np.all(tally == [65, 59, 28])
    
    print('---- ONE-SIDED ELECTION --------------------------------')
    strat1 = {'tactics' : ('minmax_preferred'),
              'frontrunnertype' : 'tally',
              'subset' : 'underdog'}
    s1 = spatial.Strategies(v).add(strat1, 0)
    e1.set_models(strategies=s1)
    result2 = e1.run(etype)
    tally = result2.runner.output['tally']
    
    print('ballots = ')
    print(result2.runner.ballots)
    print('one-sided tactical tally = ', result2.runner.output['tally'])
    assert np.all(tally == [45, 64, 25])
    
    
    
    print('---- TWO-SIDED ELECTION --------------------------------')
    strat1 = {'tactics' : ('minmax_preferred'),
              'frontrunnertype' : 'tally'}
    s1 = spatial.Strategies(v).add(strat1, 0)
    e1.set_models(strategies=s1)
    result3 = e1.run(etype)
    tally = result3.runner.output['tally']
    
    print('ballots = ')
    print(result3.runner.ballots)
    print('two-sided tactical tally = ', result3.runner.output['tally'])
    assert np.all(tally == [45, 55, 25])
