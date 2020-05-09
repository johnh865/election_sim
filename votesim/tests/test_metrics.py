# -*- coding: utf-8 -*-
import sys, traceback, pdb

import numpy as np

import votesim
from votesim.models import spatial
from votesim.metrics import ElectionStats
seed = None


def run_majority_metric(seed=0):
    
    
    
    v = spatial.SimpleVoters(seed=seed, strategy='candidate', stol=2)
    v.add_random(100)
    c = spatial.Candidates(v, seed=seed)
    c.add_random(3)
    c.add_random(2, sdev=2)
    e = spatial.Election(v, c, seed=seed)
    e.run('plurality')

    
    stats = ElectionStats(voters=v.voters,
                          candidates=c.candidates)
    
    
    # check plurality stat is correct
    stat_winner = stats.candidate.winner_plurality
    
    print('stat winner =', stat_winner)
    print('election winner=', e.winners)
    
    votecount = e.ballots.sum(axis=0)
    votecount2 = stats.candidate._winner_plurality_calcs[2]
    
    # Make sure counts for plurality are the same for election & metric
    assert np.all(votecount == votecount2)
    
    print(votecount)    
    
    
    if len(e.ties) == 0:
        assert stat_winner in e.winners
        
        maxvotes = np.max(votecount)
        numvoters = len(e.ballots)
        
        # check majority winner stat
        if maxvotes > numvoters / 2:
            s = ('there should be majority winner, '
                 '%s out of %s' % (maxvotes, numvoters))
            print(s)
            assert stats.candidate.winner_majority in e.winners
            

    
def test_majority_100():
    for i in range(100):
        run_majority_metric(i)
        print(i)
    

if __name__=='__main__':
    # test_majority_100()
    votesim.logSettings.start_debug()
    
    try:
        test_majority_100()
    except:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
    
    
    
    
    