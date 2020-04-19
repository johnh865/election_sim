# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import unittest
import numpy as np
import sys
from votesim.votesystems import tools



import votesim
from votesim.models import vcalcs

class Test_voter_rankings(unittest.TestCase):
    def test1(self):
        """Test whether cnum limit is working"""
        
        numcandidates = 20
        distances = np.random.rand(20, numcandidates)
        
        cnum1 = np.arange(20) 
        
        r1 = vcalcs.voter_rankings(None, None, cnum=None, _distances=distances)
        r2 = vcalcs.voter_rankings(None, None, cnum=cnum1, _distances=distances)

        fullranked = np.sum(r1 > 0, axis=1) == numcandidates
        
        self.assertTrue(np.all(fullranked))
        
        
        ranknum = np.sum(r2 > 0, axis=1)
        self.assertTrue(np.all(ranknum == cnum1))
        
        
class Test_zero_out_random(unittest.TestCase):
    def test1(self):
        """Test whether cnum limit is working"""
        
        
        rs = np.random.RandomState(0)
        
        numcandidates = 20
        numvoters = 100
        ratings = rs.rand(numvoters, numcandidates)
        rankings = tools.score2rank(ratings)
        climits = rs.randint(2, 5, numvoters)
        
        
        new = vcalcs.zero_out_random(rankings, climits, rs=rs)
        
        numranked = np.sum(new > 0, axis=1)
        
        self.assertTrue(np.all(climits == numranked))
        
        
    def test_weight(self):
        numcandidates = 10
        numvoters = 500        
        rs = np.random.RandomState(None)
        
        weights = [80, 10, 5, 2.5, 2.5, 0.1, 0.1, 0.1, 0, 0]
        weights = np.array(weights) / np.sum(weights)
        print('weights', weights * 100)
        
        
        ratings = rs.rand(numvoters, numcandidates)
        rankings = tools.score2rank(ratings)
        climits = np.ones(numvoters) * 2
        
        new = vcalcs.zero_out_random(rankings, climits, weights=weights, rs=rs)
        
        numranked = np.sum(new > 0, axis=0)
        print('rankings..', numranked / numvoters * 100)
        
        
class Test_distances(unittest.TestCase):
    def test1(self):
        voters = [1,2,3]
        candidates = [1,2,3]
        
        voters = np.array(voters).T
        candidates = np.array(candidates).T
        d = vcalcs.voter_distances(voters, candidates)
        print(d)
        
        correct = [[0, 1, 2],
                   [1, 0, 1],
                   [2, 1, 0],]
        correct = np.array(correct)
        self.assertTrue(np.all(d==correct))
        return
    
    def test2(self):
        voters = [[-1, 0, 2]]*5
        candidates = [[0,0,0]]*10
        
        voters = np.array(voters)
        candidates = np.array(candidates)
        d = vcalcs.voter_distances(voters, candidates)
        print(d)    
        self.assertTrue(np.all(d==3))
        
    
        
if __name__ == '__main__':
    t = Test_distances()
    t.test2()
#    unittest.main(exit=False)