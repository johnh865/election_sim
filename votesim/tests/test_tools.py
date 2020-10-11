# -*- coding: utf-8 -*-


import unittest
import numpy as np
import votesim

from votesim.votemethods import tools


class TestTools(unittest.TestCase):
    
    
    def test_winnercheck(self):
        """test winner_check function"""
        results = [1, 4, 10]
        w, t = tools.winner_check(results)
        self.assertTrue(2 in w)
        self.assertTrue(len(t) == 0)
        
        
        
        
    def test_winnercheck_ties(self):
        """test winner_check function, ties"""
        results = [10, 4, 10, 1, 2]
        w, t = tools.winner_check(results)
        self.assertIn(0, t)
        self.assertIn(2, t)
        self.assertTrue(len(w) == 0)
        
        
        w, t = tools.winner_check(results, 2)
        self.assertIn(0, w)
        self.assertIn(2, w)
        self.assertTrue(len(w) == 2)
        self.assertTrue(len(t) == 0)
        
        
    def test_winnercheck2(self):
        results = [15, 15, 10, 10, 8, 1, 2, 3]
        w, t = tools.winner_check(results, 2)
        self.assertIn(0, w)
        self.assertIn(1, w)
        self.assertTrue(len(w) == 2)
        
        w, t = tools.winner_check(results, 3)
        self.assertIn(0, w)
        self.assertIn(1, w)
        self.assertTrue(len(w) == 2)
        self.assertIn(2, t)
        self.assertIn(3, t)
        self.assertTrue(len(t) == 2)
        
        w, t = tools.winner_check(results, 4)
        self.assertIn(0, w)
        self.assertIn(1, w)    
        self.assertIn(2, w)    
        self.assertIn(3, w)    
        self.assertTrue(len(w) == 4)
        self.assertTrue(len(t) == 0)
        print('winner check!!!')
        
#        
        
    def test_getplurality(self):
        ratings = np.random.rand(500, 4)
        ranks = tools.score2rank(ratings)
        p1 = tools.getplurality(ranks=ranks)
        p2 = tools.getplurality(ratings=ratings)
        
        print('plurality generation #1')
        print(p1)
        print('plurality generation #2')
        print(p2)
        print('sums...')
        print(p1.sum(axis=0))
        print(p2.sum(axis=0))
        self.assertTrue(np.all(p1 == p2))
        
                
    
if __name__ == '__main__':
    unittest.main(exit=False)
    