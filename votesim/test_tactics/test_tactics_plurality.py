# -*- coding: utf-8 -*-
import pdb
import numpy as np

import votesim
from votesim.models import spatial
from votesim import ballot
from votesim.metrics.groups import TacticCompare

def test_plurality_bullet_preferred():
    """Test plurality bullet voting strategy.
    
    Scenario attributes:
     - 3 candidate race with 3 concentrated, coincident voter groups
     
     >>> Candidates               #0   #1   #2
     >>> Preference locations =  [-1,  0.5,  1]
     >>> Number of voters =      [ 7,    3,  5]
     
     - If voters are honest, plurality candidate #0 should win with 7 votes 
     - If voters are strategic and consider only the top two candidates, 
       Candidate #1 supporters also lean towards #2 - therefore 
       candidate #2 should win with 3 + 5 = 8 votes.
       
    """
    pref = [-1]*7 + [0.5]*3 + [1]*5
    pref = np.array(pref)[:, None]
    v = spatial.Voters(seed=0)
    v.add(pref)
    
    cpref = [-1, .5, 1]
    cpref = np.array(cpref)[:, None]
    c = spatial.Candidates(v)
    c.add(cpref)
    
    e1 = spatial.Election(voters=v, candidates=c)
    e1.run('plurality')
    tally1 = e1.result.runner.output['tally']
    
    strategy = {'tactics' : 'bullet_preferred'}
    v2 = spatial.Voters(strategy=strategy)
    v2.add(pref)
    e2 = spatial.Election(voters=v2, candidates=c)
    e2.run('plurality')
    tally2 = e2.result.runner.output['tally']
    
    # Check honest vote tally
    assert np.all(tally1 == np.array([7, 3, 5]))
    # Check strategic vote tally
    assert np.all(tally2 == np.array([7, 0, 8]))
    # Check honest winner
    assert 0 in e1.result.winners
    # Check strategic winner
    assert 2 in e2.result.winners
    return



def test_plurality_onesided():
    """Test plurality one sided voting strategy. 
    
    - #0 would win if #1 voters are strategic. 
    - #3 will win if #1 are honest and #2 voters use one-sided strategy.
    
     >>> Candidates               #0   #1    #2  #3
     >>> Preference locations =  [-1, -0.5,  0.5,  1]
     >>> Number of voters =      [ 7,    2,    3,  5]
     
     
    """
    # Generate voter preferences
    pref = [-1]*7 + [-0.5]*2 + [0.5]*3 + [1]*5
    pref = np.array(pref)[:, None]
    v = spatial.Voters(seed=0)
    v.add(pref)
    
    # Generate candidate preferences    
    cpref = [-1, -.5, .5, 1]
    cpref = np.array(cpref)[:, None]
    c = spatial.Candidates(v)
    c.add(cpref)
    
    # Run honest election
    e1 = spatial.Election(voters=v, candidates=c)
    e1.run('plurality')
    tally1 = e1.result.runner.output['tally']
    assert np.all(tally1 == np.array([7, 2, 3, 5]))
    assert 0 in e1.result.winners
    
    # Run one-sided tactical election
    strategy = {'tactics' : 'bullet_preferred', 'onesided' : True}
    v2 = spatial.Voters(strategy=strategy)
    v2.add(pref)
    e2 = spatial.Election(voters=v2, candidates=c)
    e2.run('plurality')
    tally2 = e2.result.runner.output['tally']    
    assert np.all(tally2 == np.array([7, 2, 0, 8]))
    assert 3 in e2.result.winners
    
    # Test metric comparison system.
    tc = TacticCompare(e_strat=e2.electionStats,
                       e_honest=e1.electionStats,
                       )
    # e2.electionStats.add_output(tc)

    print('one-sided regret change =', tc.regret)    
    print('one-sided VSE change = ', tc.regret_efficiency_candidate)
    print('VSE tactical  = ', e2.electionStats.winner.regret_efficiency_candidate)
    print('VSE honest  = ', e1.electionStats.winner.regret_efficiency_candidate)
    
    
    
    # Run full tactical election
    strategy = {'tactics' : 'bullet_preferred', 'onesided' : False}
    v3 = spatial.Voters(strategy=strategy)
    v3.add(pref)
    e3 = spatial.Election(voters=v3, candidates=c)
    e3.run('plurality')
    tally3 = e3.result.runner.output['tally']    
    assert np.all(tally3 == np.array([9, 0, 0, 8]))
    assert 0 in e3.result.winners
    
    
    # Test metric comparison system.
    tc = TacticCompare(e_strat=e3.electionStats,
                       e_honest=e1.electionStats,
                       )
    # e3.electionStats.add_output(tc)
    print('')
    print('two-sided regret change =', tc.regret)    
    print('two-sided VSE change = ', tc.regret_efficiency_candidate)
    
    docs = e3.result.output_docs

    # Try to append new output to election results
    e3.result.append_stat(tc)
    df = e3.dataframe()

    # pdb.set_trace()
    return df

if __name__ == '__main__':
    test_plurality_bullet_preferred()
    tc = test_plurality_onesided()
    
    
    
    
    
    
    
    
    
    
    
    
    