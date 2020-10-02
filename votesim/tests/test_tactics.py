# -*- coding: utf-8 -*-
"""
Test simple plurality scenario.

 - Four candidates
 - Voters in groups of 3, 4, and 2
 - Preference distances of:
     
   >>>  Candidates
   >>>  #0 #1 #2 #3  
   >>> [[0, 5, 5, 5]]*3 + 
   >>> [[5, 5, 5, 0]]*4 + 
   >>> [[1, 0, 5, 5]]*2    
   
 - Honest votes are
 
   >>>  Candidates
   >>>  #0 #1 #2 #3  
   >>> [[1, 0, 0, 0]]*3 + 
   >>> [[0, 0, 0, 1]]*4 + 
   >>> [[0, 1, 0, 0]]*2
   
   For strategy the last group ought to change their votes from #1 to #0
   
"""
import numpy as np
import pandas as pd
import sys
from votesim.models import spatial
from votesim import ballot

# v = spatial.Voters(0)
# v.add_random(5, 3)
# c = spatial.Candidates(v, 0)
# c.add_random(3)

# e = spatial.Election(voters=v, candidates=c)
# e.user_data(a=0, b=1)


# ballots = e.vballots.ballots(etype)
# tact_ballot = ballot.TacticalBallots(etype, ballots)


class TestPluralityTactics(object):
    def __init__(self):
        etype = 'plurality'

        distances = (
                     [[0, 5, 5, 5]]*3 + 
                     [[5, 5, 5, 0]]*4 + 
                     [[1, 0, 5, 5]]*2
                     )

        distances = np.array(distances, dtype=float)
        b = ballot.BaseBallots(distances=distances)
        b = b.rate_linear().rate_norm().rank_honest()
        t = ballot.TacticalBallots(etype, ballots=b)
        
        self.ballots = t
        assert t.votes[7, 1] == 1
        assert t.votes[8, 1] == 1
        
        
    def test_frontrunners(self):
        assert 0 in self.ballots.front_runners
        assert 3 in self.ballots.front_runners
        return
                
            
    def test_compromise(self):
        t = self.ballots.copy()
        t = t.compromise()
        assert t.votes[7, 0] == 1
        assert t.votes[8, 0] == 1
        assert np.all(t.votes[7, 1:] == 0)
        assert np.all(t.votes[8, 1:] == 0)

        return


# class TestRankTactics(object):
    

if __name__ == '__main__':
    t = TestPluralityTactics()
    t.test_compromise()
    t.test_frontrunners()