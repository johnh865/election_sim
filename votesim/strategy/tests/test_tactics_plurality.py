# -*- coding: utf-8 -*-
"""Basic tests for plurality voting tactical simulations. 

Test one-sided and fully tactical elections. 

"""
import pdb
import numpy as np

import votesim
from votesim.models import spatial
from votesim.strategy import TacticalBallots
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
    
    #####################################################################
    # Run honest election    
    
    e1 = spatial.Election(voters=v, candidates=c)
    e1.run('plurality')
    tally1 = e1.result.runner.output['tally']
    result1 = e1.result
    #####################################################################
    # Run strategic election
  
    strategy1 = {'tactics' : 'bullet_preferred',
                 'ratio' : 1,
                 'subset' : 'underdog',
                 'underdog' : None,}
    s = spatial.Strategies(v)
    s.add(**strategy1)
    
    
    e1.set_models(strategies=s)
    e1.run('plurality', result=result1)
    tally2 = e1.result.runner.output['tally']
    result2 = e1.result
    # Check honest vote tally
    assert np.all(tally1 == np.array([7, 3, 5]))
    # Check strategic vote tally
    assert np.all(tally2 == np.array([7, 0, 8]))
    # Check honest winner
    assert 0 in result1.winners
    # Check strategic winner
    assert 2 in result2.winners
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
    
    #####################################################################
    # Run honest election
    e1 = spatial.Election(voters=v, candidates=c)
    e1.run('plurality')
    tally1 = e1.result.runner.output['tally']
    result1 = e1.result
    
    assert np.all(tally1 == np.array([7, 2, 3, 5]))
    assert 0 in e1.result.winners
    
    #####################################################################
    # Run one-sided tactical election
    
    strategy1 = {'tactics' : 'bullet_preferred', 
                 'ratio' : 1,
                 'subset' : 'underdog',
                 'underdog' : None,}
    strat1 = spatial.Strategies(v).add(**strategy1)
    
    
    e2 = spatial.Election(voters=v, candidates=c, strategies=strat1)
    result2 = e2.run('plurality', result=result1)
    tally2 = result2.runner.output['tally']    
    assert np.all(tally2 == np.array([7, 2, 0, 8]))
    assert 3 in result2.winners
    
    # Test metric comparison system.
    tc = TacticCompare(e_strat=result2.stats,
                       e_honest=result1.stats,
                       )
    # e2.electionStats.add_output(tc)
    stats2 = result2.stats
    stats1 = result1.stats
    print('---------------------------------------')
    print('one-sided regret change =', tc.regret)    
    print('')
    print('one-sided VSE change = ', tc.regret_efficiency_candidate)
    print('')
    print('VSE tactical  = ', stats2.winner.regret_efficiency_candidate)
    print('VSE honest  = ', stats1.winner.regret_efficiency_candidate)
    
    #####################################################################
    # Run full tactical election
    
    strategy1 = {'tactics' : 'bullet_preferred',
                 'ratio' : 1,
                 'underdog' : None,
                 'subset' : ''}
    strat1 = spatial.Strategies(v).add(**strategy1)
    
    e3 = spatial.Election(voters=v, candidates=c, strategies=strat1)
    result3 = e3.run('plurality', result=result1)
    tally3 = result3.runner.output['tally']    
    assert np.all(tally3 == np.array([9, 0, 0, 8]))
    assert 0 in result3.winners
    
    # Test metric comparison system.
    tc = TacticCompare(e_strat=result3.stats,
                       e_honest=result1.stats,
                       )
    print('')
    print('two-sided regret change =', tc.regret)    
    print('two-sided VSE change = ', tc.regret_efficiency_candidate)
    
    docs = result3.output_docs

    # Try to append new output to election results
    e3.append_stat(tc)
    df = e3.dataframe()

    # pdb.set_trace()
    return df



def test_plurality_chain():
    """Test re-using honest election data to initialize 
    strategic runs."""
    
    pref = [-1]*7 + [-0.5]*2 + [0.5]*3 + [1]*5
    pref = np.array(pref)[:, None]
    v = spatial.Voters(seed=0)
    v.add(pref)
    
    # Generate candidate preferences    
    cpref = [-1, -.5, .5, 1]
    cpref = np.array(cpref)[:, None]
    c = spatial.Candidates(v)
    c.add(cpref)

    # Run honest
    e1 = spatial.Election(voters=v, candidates=c)
    e1.run('plurality',)
    tally1 = e1.result.runner.output['tally']
    result1 = e1.result
    assert np.all(tally1 == np.array([7, 2, 3, 5]))
    
    # Run tactical
    strategy = {'tactics' : 'bullet_preferred',
                 'ratio' : 1,
                 'underdog' : None,
                 'subset' : ''
                 }
    s1 = spatial.Strategies(v).add(**strategy)
    e1.set_models(strategies=s1)
    e1.run('plurality', result=result1)
    tally2 = e1.result.runner.output['tally']
    assert np.all(tally2 == np.array([9, 0, 0, 8]))
    
    # Run one sided
    strategy = {'tactics' : 'bullet_preferred',
                 'ratio' : 1,
                 'underdog' : None,
                 'subset' : 'underdog'}
    s1 = spatial.Strategies(v).add(**strategy)
    e1.set_models(strategies=s1)
    e1.run('plurality', result=result1)
    tally3 = e1.result.runner.output['tally']
    assert np.all(tally3 == np.array([7, 2, 0, 8]))
    
    return locals()


def test_plurality_ratio():
    """Test adjusted the ratio of tactical to honest voters"""
    
    v = spatial.Voters(seed=0)
    v.add_random(100)
    c = spatial.Candidates(voters=v, seed=1)
    c.add_random(5)
    
    e1 = spatial.Election(voters=v, candidates=c)
    result1 = e1.run('plurality')
    tally1 = e1.result.runner.output['tally']
    print('---------------------------------------')
    for tacti_num in [0, 10, 25, 50, 75, 100]:
        ratio = tacti_num / 100.
        strategy =  {'tactics' : 'bullet_preferred', 
                     'ratio' : ratio,
                     'underdog' : None,
                     'subset' : ''}
        s1 = spatial.Strategies(v).add(**strategy)
        e1.set_models(strategies=s1)
        
        e1.run('plurality', result=result1)
        tally2 = e1.result.runner.output['tally']
        print('vote tally =', tally2)
        bgen = e1.ballotgen
        tactical_ballots = bgen.tacticalballots
        num_tactical_voters = len(tactical_ballots.group_index['tactical-0'])
        
        # check that tactical voters number is correct
        assert num_tactical_voters == tacti_num
        
        # check that total voters in group index is correct. 
        group_index = tactical_ballots.group_index
        count1 = len(group_index['honest-0'])
        count2 = len(group_index['tactical-0'])
        assert count1 + count2 == 100
        count3 = len(group_index['topdog-0'])
        count4 = len(group_index['underdog-0'])
        assert count3 + count4 == count2
    return





if __name__ == '__main__':
    test_plurality_bullet_preferred()
    df = test_plurality_onesided()
    l = test_plurality_chain()
    test_plurality_ratio()
    
    
    
    
    
    
    
    
    
    
    
    
    