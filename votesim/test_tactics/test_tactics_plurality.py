# -*- coding: utf-8 -*-
import numpy as np

import votesim
from votesim.models import spatial
from votesim import ballot


def test_plurality_bullet_preferred():
    """Test plurality bullet voting strategy.
    
    Scenario attributes:
     - 3 candidate race with 3 concentrated, coincident voter groups
     
     >>> Candidates               #0   #1   #2
     >>> Preference locations at [-1,  0.5,  1]
     >>> Number of voters at     [ 7,    3,  5]
     
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
    
    
    assert np.all(tally1 == np.array([7, 3, 5]))
    assert np.all(tally2 == np.array([7, 0, 8]))
    assert 0 in e1.result.winners
    assert 2 in e2.result.winners
    return


def ranked_setup()

def test_ranked_burial():
    """
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
    
    Burial strategies
    --------------------------
    * For #0 voters, their ballots are unchanged as their least favorite is 
      already buried.
    * For #1 voters, they decide to bury #2.
    * For #2 voters, they decide to bury #1. 
    
    
    Outcomes
    --------
    Burial strategies backfires with #1 and #2 voters, allowing #0 to win. 
    
    Returns
    -------
    None.

    """
    pref = [-1]*7 + [0.5]*3 + [1]*5
    pref = np.array(pref)[:, None]
    v = spatial.Voters(seed=0)
    v.add(pref)
    
    cpref = [-1, .5, 1]
    cpref = np.array(cpref)[:, None]
    c = spatial.Candidates(v)
    c.add(cpref)
    
    # run honest election
    e1 = spatial.Election(voters=v, candidates=c)
    e1.run('ranked_pairs')
    
    # Make sure condorcet winner was found
    assert 1 in e1.result.winners
    
    #run strategic election
    strategy = {'tactics' : 'bury'}
    v2 = spatial.Voters(strategy=strategy)
    v2.add(pref)    
    e2 = spatial.Election(voters=v2, candidates=c)
    e2.run('ranked_pairs')    
    
    # Make sure the correct front runners were found
    assert 1 in e2.tactical_ballots.front_runners
    assert 2 in e2.tactical_ballots.front_runners
    
    
    return
    

if __name__ == '__main__':
    test_ranked_burial()
    
    
    
    
    
    
    
    
    
    
    
    
    