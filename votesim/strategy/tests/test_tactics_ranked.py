# -*- coding: utf-8 -*-
"""
Scenario attributes:
  - 3 candidate race with 3 concentrated, coincident voter groups
  
  >>> Candidates               #0   #1   #2
  >>> Preference locations at [-1,  0.5,  1]
  >>> Number of voters at     [ 7,    3,  5]    
 
 Rankings
 ---------
 >>> Candidates  #0 #1 #2
 >>> Ranks       [1, 2, 3] x 7 voters
 >>> Ranks       [3, 1, 2] x 3 voters
 >>> Ranks       [3, 2, 1] x 5 voters

 Condorcet winner is candidate #1
 
 Eliminating #1....
 
 >>> ranks [1, -, 2] x 7 voters
 >>> ranks [2, -, 1] x 3 voters
 >>> ranks [2, -, 1] x 5 voters
 
 Condorcet runner up is candidate #2
"""
import numpy as np
import pytest

import votesim
from votesim.models import spatial
from votesim import ballot

voter_pref = [-1]*7 + [0.5]*3 + [1]*5
voter_pref = np.array(voter_pref)[:, None]
candidate_pref = [-1, .5, 1]
candidate_pref = np.array(candidate_pref)[:, None]


def test_ranked_deep_bury():
    """Test ranked burial tactic. 
    
    Burial strategies
    --------------------------
    * For #0 voters, their ballots are unchanged as their least favorite is 
      already buried.
    * For #1 voters, they decide to bury #2.
    * For #2 voters, they decide to bury #1. 
    
    Outcomes
    --------
    Burial strategies backfires with #1 and #2 voters, allowing #0 to win. 
    """
    v = spatial.Voters(seed=0)
    v.add(voter_pref)
    c = spatial.Candidates(v)
    c.add(candidate_pref)
    
    # run honest election
    e1 = spatial.Election(voters=v, candidates=c)
    e1.run('ranked_pairs')
    
    # Make sure condorcet winner was found
    assert 1 in e1.result.winners
    
    #run strategic election
    strategy = {'tactics' : 'deep_bury', 'frontrunnertype' : 'eliminate'}
    v2 = spatial.Voters(strategy=strategy)
    v2.add(voter_pref)    
    e2 = spatial.Election(voters=v2, candidates=c)
    e2.run('ranked_pairs')    
    
    # Make sure the correct front runners were found
    assert 1 in e2.used_ballots.front_runners
    assert 2 in e2.used_ballots.front_runners
    
    # Check that #0 is the winner
    assert 0 in e2.result.winners
    
    ballots = e2.result.ballots
    ballots = votesim.votemethods.tools.rcv_reorder(ballots)
    
    # Check the new tactical rankings
    right = [
       [1, 2, 0],
       [1, 2, 0],
       [1, 2, 0],
       [1, 2, 0],
       [1, 2, 0],
       [1, 2, 0],
       [1, 2, 0],
       [3, 1, 0],
       [3, 1, 0],
       [3, 1, 0],
       [3, 0, 1],
       [3, 0, 1],
       [3, 0, 1],
       [3, 0, 1],
       [3, 0, 1]]
    right = votesim.votemethods.tools.rcv_reorder(right)
    
    assert np.all(right == ballots)
    return


def test_ranked_deep_bury_onesided():
    """Test one sided burial strategy.
    
    For one-sided, only the under-dog voters vote tactically. 
    
    * Honest Condorcet winner is Candidate #1. 
    * Runner up is Candidate #2. 
    * Therefore only #2 voters vote strategically in this scenario. 
    

    Outcomes
    -------
    It seems like burial backfires again in this scenario. 

    """
    v = spatial.Voters(seed=0)
    v.add(voter_pref)
    c = spatial.Candidates(v)
    c.add(candidate_pref)    
    
    strategy = {'tactics' : 'deep_bury',
                'subset' : 'underdog',
                'frontrunnertype' : 'eliminate'}
    v2 = spatial.Voters(strategy=strategy)
    v2.add(voter_pref)    
    e2 = spatial.Election(voters=v2, candidates=c)
    e2.run('ranked_pairs')    
    
    right = [
       [1, 2, 3],
       [1, 2, 3],
       [1, 2, 3],
       [1, 2, 3],
       [1, 2, 3],
       [1, 2, 3],
       [1, 2, 3],
       [3, 1, 2],
       [3, 1, 2],
       [3, 1, 2],
       [3, 0, 1],
       [3, 0, 1],
       [3, 0, 1],
       [3, 0, 1],
       [3, 0, 1]]
    right = votesim.votemethods.tools.rcv_reorder(right)
    ballots = e2.result.ballots
    ballots = votesim.votemethods.tools.rcv_reorder(ballots)
    assert np.all(right == ballots)    
    return


def test_ranked_bury():
    
    
    v = spatial.Voters(seed=0)
    v.add(voter_pref)
    c = spatial.Candidates(v)
    c.add(candidate_pref)    
    
    strategy = {'tactics' : 'bury',
                'subset' : '',
                'frontrunnertype' : 'eliminate'}
    v2 = spatial.Voters(strategy=strategy)
    v2.add(voter_pref)    
    e2 = spatial.Election(voters=v2, candidates=c)
    e2.run('ranked_pairs')    
    
    right = [
       [1, 2, 0],
       [1, 2, 0],
       [1, 2, 0],
       [1, 2, 0],
       [1, 2, 0],
       [1, 2, 0],
       [1, 2, 0],
       [0, 1, 0],
       [0, 1, 0],
       [0, 1, 0],
       [0, 0, 1],
       [0, 0, 1],
       [0, 0, 1],
       [0, 0, 1],
       [0, 0, 1]]
    right = votesim.votemethods.tools.rcv_reorder(right)
    ballots = e2.result.ballots
    ballots = votesim.votemethods.tools.rcv_reorder(ballots)
    assert np.all(right == ballots)      
    return


    

if __name__ == '__main__':
    test_ranked_deep_bury()
    test_ranked_deep_bury_onesided()
    test_ranked_bury()
    
    
    
    
    
    
    
    
    
    
    
    
    