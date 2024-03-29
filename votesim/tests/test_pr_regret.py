# -*- coding: utf-8 -*-

import unittest
import numpy as np
from votesim.metrics import PrRegret, consensus_regret, ElectionStats
from votesim.models.spatial import Voters, Candidates
from votesim.models.dataclasses import VoterData, CandidateData, ElectionData


class Test_PR_Regret(unittest.TestCase):
    
    
    def test_perfect(self):
        """Test case where people perfectly represented"""
        # Test equal spaced scenario
        # 15 voters of preferences -1, 0, and 1
        voters = [-1]*5 + [0]*5 + [1]*5
        voters = np.atleast_2d(voters).T
        
        # Candidates with same preferences as voters
        winners = [-1, 0, 1]
        winners = np.atleast_2d(winners).T
        
        
        e = ElectionStats()
        e.set_raw(voters=voters,
                  candidates=winners, 
                  winners=[0, 1, 2])
        pr = PrRegret(e)
        
        self.assertEqual(pr.avg_regret, 0)
        self.assertEqual(pr.winners_regret_std, 0)
        
        
    def test_compare(self):
        """PR regret and consensus regret ought to produce the same result
        when only one winner"""
        
        v = Voters(0)
        v.add_random(100,4).build()
        
        c = Candidates(v, seed=0)        
        c.add_random(cnum=5).build()
        
        e = ElectionStats()
        e.set_raw(voters=v.data.pref,
                  candidates=c.data.pref,
                  winners=[0])

        p = PrRegret(e)
        r1  = p.avg_regret
        r2 = consensus_regret(v.data.pref, c.data.pref[0:1])
        
        print('PR regret =', r1)
        print('consensus regret =', r2)
        self.assertTrue(np.round((r1-r2) / r1, 4) == 0)
        
        
    def test_nearest_regrets(self):
        """The nearest regrets for each candidate ought to add up to the total
        PR regret"""
        
        v = Voters(0)
        v.add_random(100,4).build()
        
        c = Candidates(v, 0)        
        c.add_random(cnum=5).build()
        
        e = ElectionStats(voters=v.data, candidates=c.data)
        
        e.set_raw(voters=v.data.pref,
                  weights=None,
                  order=v.data.order,
                  candidates=c.data.pref,
                  winners=np.arange(5),
                  )
        
        pr = PrRegret(e)
        r1 = pr.avg_regret
        r2s = pr.winners_regret
        r2 = np.sum(r2s)
        
        print('PR regret =', r1)
        print('nearest regrets =', r2s)        
        self.assertTrue(np.round((r1-r2) / r1, 4) == 0)


        
        
    
if __name__ == '__main__':
    # unittest.main(exit=False)
    t = Test_PR_Regret()
    t.test_perfect()
    t.test_compare()
    t.test_nearest_regrets()
    