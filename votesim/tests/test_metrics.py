# -*- coding: utf-8 -*-
"""Test to see if the winner_majority metric is working correctly."""
import sys, traceback, pdb

import numpy as np

import votesim
from votesim.models import spatial
from votesim.metrics import ElectionStats
seed = None


def run_majority_metric(seed=0):
    
    
    v = spatial.Voters(seed=seed, tol=None)
    v.add_random(100)
    c = spatial.Candidates(v, seed=seed)
    c.add_random(3)
    c.add_random(2, sdev=2)
    e = spatial.Election(v, c, seed=seed)
    e.run('plurality')
    
    # Retrieve output
    ties = e.result.ties
    winners = e.result.winners
    ballots = e.result.ballots
    

    
    stats = ElectionStats(voters=v.data,
                          candidates=c.data)
    
    
    # check plurality stat is correct
    stat_winner = stats.candidates.winner_plurality
    
    print('stat winner =', stat_winner)
    print('election winner=', winners)
    
    votecount = ballots.sum(axis=0)
    votecount2 = stats.candidates._winner_plurality_calcs[2]
    
    # Make sure counts for plurality are the same for election & metric
    assert np.all(votecount == votecount2)
    
    print(votecount)    
    
    
    if len(ties) == 0:
        assert stat_winner in winners
        
        maxvotes = np.max(votecount)
        numvoters = len(ballots)
        
        # check majority winner stat
        if maxvotes > numvoters / 2:
            s = ('there should be majority winner, '
                 '%s out of %s' % (maxvotes, numvoters))
            print(s)
            assert stats.candidates.winner_majority in winners
            

    
def test_majority_100():
    for i in range(100):
        run_majority_metric(i)
        print(i)
    

if __name__=='__main__':
    test_majority_100()
    
    # try:
    #     test_majority_100()
    # except:
    #     extype, value, tb = sys.exc_info()
    #     traceback.print_exc()
    #     pdb.post_mortem(tb)
    
    
    
    
    