# -*- coding: utf-8 -*-

import unittest
import numpy as np
from votesim.votesystems import condorcet

class TestSmith(unittest.TestCase):
    
    def test_smithset1(self):
        #     M  N  C  K
        d = [[  0,  16, -20, -7, 2],
             [-16,   0,  13, 17, 2],
             [ 20, -13,   0,  7, 2],
             [  7, -17,  -7,  0, 2],
             [ -2,  -2,  -2, -2, 0]]
             
            
        d = np.array(d)
        r = condorcet.smith_set(matrix=d)
        print('smith set')
        print(r)
        solution = np.array([0, 1, 2, 3])
        r = np.sort(list(r))
        self.assertTrue(np.all(solution == r))
        
    def test_smith_minimax1(self):
        d = [[  0,  16,  20, -7, 2],
             [-16,   0,  13, 17, 2],
             [-20, -13,   0,  7, 2],
             [  7, -17,  -7,  0, 2],
             [ -2,  -2,  -2, -2, 0]]
             
            
        d = np.array(d)
        w, t = condorcet.smith_minimax(matrix=d)
        self.assertTrue(0 in w)
        self.assertTrue(len(t) == 0)
        return
    
    def test_smith_minimax2(self):
        d = [[  0,  16, -20, -7, 2],
             [-16,   0,  13, 17, 2],
             [ 20, -13,   0,  7, 2],
             [  7, -17,  -7,  0, 2],
             [ -2,  -2,  -2, -2, 0]]
             
            
        d = np.array(d)
        w, t = condorcet.smith_minimax(matrix=d)
        self.assertTrue(2 in w)
        self.assertTrue(len(t) == 0)
        return    
        
    
    def test_smith_minimax3(self):
        d = [[  0,  16,  20, -7,  2],
             [-16,   0,  13,  3,  2],
             [-20, -13,   0,  2,  2],
             [  7,  -3,  -2,  0,  2],
             [ -2,  -2,  -2, -2,  0]]
             
            
        d = np.array(d)
        w, t = condorcet.smith_minimax(matrix=d)
        self.assertTrue(3 in w)
        self.assertTrue(len(t) == 0)
        return    
    
    def test_ranked_condorcet(self):
        #     M  N  C  K
        d = [[1, 2, 3, 4]]*42 + \
            [[4, 1, 2, 3]]*26 + \
            [[4, 3, 1, 2]]*15 + \
            [[4, 3, 2, 1]]*17
            
        d = np.array(d)
        i = condorcet.ranked_condorcet(d)
        self.assertEqual(i, 1)
        return
        
    
    def tests_minimax_condorcet(self):
        """Make sure smith_minimax successfully returns condorcet winner"""
        for seed in range(100):
            rs = np.random.RandomState(None)
                
            d = rs.randint(1, 5, size=(4,4))
            i1 = condorcet.ranked_condorcet(d)
            if i1 != -1:
                w,t, *args = condorcet.smith_minimax(ranks=d)
                self.assertEqual(w[0], i1)
                print('seed %d, winner=%d' % (seed, i1))
            else:
                print('no condorcet winner')
        return
    
    
        
#    def test_wiki(self):
#        """
#        Test example from wikipedia, retrieved Dec 19, 2019.
#        Correct results taken from wikipedia (winner knoxville K)
#        https://en.wikipedia.org/wiki/Instant-runoff_voting
#        """
#        
#        #     M  N  C  K
#        d = [[1, 2, 3, 4]]*42 + \
#            [[4, 1, 2, 3]]*26 + \
#            [[4, 3, 1, 2]]*15 + \
#            [[4, 3, 2, 1]]*17
#            
#        d = np.array(d)
#        winners, ties, history = condorcet.ranked_pairs(d, 1)
##        print('test wiki')
##        print(winners)
##        print(history)
##        
##        correct_history = [[42, 26, 15, 17],
##                           [42, 26,  0, 32],
##                           [42,  0,  0, 58]]
##        
##        correct_history = np.array(correct_history)
#            
#        self.assertTrue(np.all(correct_history == history))
#        self.assertEqual(winners[0], 3)
    
if __name__ == '__main__':
    unittest.main(exit=False)
    
    
    
    
    
    