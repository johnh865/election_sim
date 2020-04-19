# -*- coding: utf-8 -*-

import unittest
import numpy as np
from votesim.metrics import PrRegret, consensus_regret
from votesim.models.spatial import SimpleVoters, Candidates



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
        
        p = PrRegret(voters, winners)
        p.regret
        
        self.assertEqual(p.regret, 0)
        self.assertEqual(p.std_voters, 0)
        
        
    def test_compare(self):
        """PR regret and consensus regret ought to produce the same result
        when only one winner"""
        
        v = SimpleVoters(0)
        v.add_random(100,4)
        
        c = Candidates(v, 0)        
        c.add_random(cnum=5)
        
        p = PrRegret(v.voters, c.candidates[0:1])
        r1  = p.regret
        r2 = consensus_regret(v.voters, c.candidates[0:1])
        
        print('PR regret =', r1)
        print('consensus regret =', r2)
        self.assertTrue(np.round((r1-r2) / r1, 4) == 0)
        
        
    def test_nearest_regrets(self):
        """The nearest regrets for each candidate ought to add up to the total
        PR regret"""
        
        v = SimpleVoters(0)
        v.add_random(100,4)
        
        c = Candidates(v, 0)        
        c.add_random(cnum=5)
        
        p = PrRegret(v.voters, c.candidates)
        r1 = p.regret
        r2s = p.nearest_regrets
        r2 = np.sum(r2s)
        
        print('PR regret =', r1)
        print('nearest regrets =', r2s)        
        self.assertTrue(np.round((r1-r2) / r1, 4) == 0)


        
        
    
if __name__ == '__main__':
    unittest.main(exit=False)
    