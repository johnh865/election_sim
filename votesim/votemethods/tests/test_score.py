# -*- coding: utf-8 -*-

import unittest
import numpy as np
import votesim

from votesim.votemethods import score


class TestScore(unittest.TestCase):
    
    def test_result(self):
        """https://www.rangevoting.org/RRV.html"""
        
        print('\nTEST REWEIGHTED RANGE #1')
        d = 50*[[ 0, 42, 99]] + \
            50*[[43, 99,  0]] + \
            40*[[99,  0, 53]] 
            
        # First round, index=2 is winner. 
        # Total scores of winners are..
        # 50 * [[99]] + \
        # 50 * [[0]] + \
        # 40 * [[53]]
        
        # New weight is C / (C + SUM)
        # 99 / (99 + 99) = 1/2.
        # 99 / (99 + 0) = 1.
        # 99 / (99 + 53) = 0.651315
        
        # Reweighted scores = 
        # 50*[[0.0, 21.5, 0 ]] + \
        # 50*[[43 , 99  , 0    ]] + \
        # 40*[[64.48, 0 , 0]]
        # ----------------------
        # new scores = [4729.2, 6000, 0]
            
        d = np.array(d)
        w, t, hist = score.reweighted_range(d,
                                            numwin=2,
                                            C_ratio=1, 
                                            maxscore=99)
        hist = np.round(hist)
        
        # specify correct answer from warren smith's website
        hist_right = [[6110, 7050, 7070],
                      [4729, 6000, 0   ]]
        
        winners_right = [2, 0]
        
        hist_right = np.array(hist_right)
        winners_right = np.array(winners_right)
        
        print('winners', w)
        print('history', hist)
        
        self.assertTrue(0 in winners_right)
        self.assertTrue(2 in winners_right)
        self.assertTrue(len(winners_right) == 2)
        self.assertTrue(np.all(hist == hist_right))
        return 
    
    
    
    def test_result2(self):
        """https://www.rangevoting.org/RRVr.html"""
        print('\nTEST REWEIGHTED RANGE #2')
        d = [[10, 9, 8, 1, 0]] * 60 + [[0, 0, 0, 10, 10]] * 40
        d = np.array(d)
        
        # Call the STV function
        w, t, h = score.reweighted_range(d, 3,
                                         C_ratio=1,
                                         maxscore=10)
        # results from
        # https://www.rangevoting.org/RRVr.html
        htrue = [[600, 540, 480, 460, 400],
                 [  0, 270, 240, 430, 400],
                 [  0, 257, 229,   0, 200]]
        
        wtrue = [0, 3, 1]
        
        w = np.round(w)
        h = np.round(h)
        
        print('winners', w)
        print('history', h)
        self.assertTrue(np.all(w == wtrue))
        self.assertTrue(np.all(h == htrue))    
        
        
class TestStar(unittest.TestCase):
    def test_tie3(self):
        print('Test STAR winners check')
        d = [[0 ,1, 2],
             [0, 2, 1],
             [3, 0, 0]]
        d = np.array(d)
        w, t, out = score.star(d)
        self.assertIn(1, t)
        self.assertIn(2, t)
        self.assertTrue(len(w) == 0)
        
        
    def test_wiki(self):
        """test scenario in wikipedia 
        https://en.wikipedia.org/wiki/STAR_voting
        retrieved 25-Dec 2019
        """
        
        # M N C K
        d = [[5, 2, 1, 0]]*42 + \
            [[0, 5, 2, 1]]*26 + \
            [[0, 3, 5, 3]]*15 + \
            [[0, 2, 4, 5]]*17
        d = np.array(d)
        w, t, d = score.star(d)
        for k, v in d.items(): 
            print(k + ':')
            print(v)
            
        runoff_candidates = d['runoff_candidates']
        sums = d['tally'] 
        self.assertIn(1, runoff_candidates)
        self.assertIn(2, runoff_candidates)
        self.assertIn(1, w)
        self.assertTrue(len(t) == 0)
        self.assertTrue(np.all(sums == [210, 293, 237, 156]))
    
        
class TestSequentialMonroe(unittest.TestCase):
    def test_result(self):
        
        print('\nTEST SEQUENTIAL MONROE')
        d = 50*[[ 0, 42, 99]] + \
            50*[[43, 99,  0]] + \
            40*[[99,  0, 53]] 
            
            
        w, t, h = score.sequential_monroe(d, numwin=2)
        print('sequential monroe')
        print('winners', w)
        print('ties', t)
        print('history\n', h)
        
        self.assertIn(1, w)
        self.assertIn(2, w)
#    
if __name__ == '__main__':
    unittest.main(exit=False)
    