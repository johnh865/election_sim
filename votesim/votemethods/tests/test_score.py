# -*- coding: utf-8 -*-

import unittest
import numpy as np
import votesim

from votesim.votemethods import score
from votesim.models import spatial

class TestRRW(unittest.TestCase):
    
    def test_rrw1(self):
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
        hist = np.round(hist['round_history'])
        
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
    
    
    
    def test_rrw2(self):
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
        h = np.round(h['round_history'])
        
        print('winners', w)
        print('history', h)
        self.assertTrue(np.all(w == wtrue))
        self.assertTrue(np.all(h == htrue))    
        
        
class TestStar(unittest.TestCase):
    def test_tie(self):
        print('Test STAR winners check')
        d = [[0 ,1, 2],
             [0, 2, 1],
             [3, 0, 0]]
        d = np.array(d)
        w, t, out = score.star(d)
        self.assertIn(1, t)
        self.assertIn(2, t)
        self.assertTrue(len(w) == 0)
        
        
    
    def test_tie3(self):
        """Test score tie for runner ups."""
        d = [[0, 2, 2, 4],
             [0, 1, 1, 3],
             [0, 1, 1, 0]]
        
        d = np.array(d)
        w, t, o = score.star(d)
        return
    
    
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
        
        
    def test_zeros(self):
        
        d = np.zeros((101, 3))
        w, t, d = score.star5(d, 1)
        self.assertTrue(len(w) == 0)
        print('success')
        return
    
    
    def test_tally(self):
        np.random.seed(0)
        data = np.random.randint(0, 5, size=(100, 4))
        w, t, d = score.star(data)
        
        self.assertTrue(len(w) == 1)
        self.assertTrue(w[0] == 3)
        return
    
    
    def test_multiwinner(self):
        np.random.seed(0)
        data = np.random.randint(0, 5, size=(50, 6))
        winners, ties, outputs = score.star(data, numwin=3)
        winners0, ties0, outputs0 = score.star(data, numwin=1)
        
        assert len(winners) + len(ties) >= 3
        assert 'elimination_rounds' in outputs
        assert hasattr(outputs, 'keys')
        assert hasattr(outputs, 'values')
        assert winners0 == winners[0]
        
        return
        
    
        
    
        
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
        
        self.assertIn(0, w)
        self.assertIn(2, w)
        
        
    def test_spatial(self):
        
        numvoters = 99
        num_candidates = 20
        for ii in range(50):
            v = spatial.Voters(seed=ii,)
            v.add_random(numvoters=numvoters, ndim=2, )
            c = spatial.Candidates(voters=v, seed=0)
            c.add_random(cnum=num_candidates, sdev=1.0)
            e = spatial.Election(voters=v, candidates=c)
            scores = e.ballotgen.get_honest_ballots(
                etype=votesim.votemethods.SCORE
            )
            winners, ties, output = score.sequential_monroe(scores, numwin=6)
        return
    
    
    def test_score_compare(self):
        """Sequential monroe ought to devolve to score at numwinners=1"""
        numvoters = 50
        num_candidates = 6
        for ii in range(50):
            v = spatial.Voters(seed=ii,)
            v.add_random(numvoters=numvoters, ndim=2, )
            c = spatial.Candidates(voters=v, seed=0)
            c.add_random(cnum=num_candidates, sdev=1.0)
            e = spatial.Election(voters=v, candidates=c)
            scores = e.ballotgen.get_honest_ballots(
                etype=votesim.votemethods.SCORE
            )
        
            winners1, ties1, output = score.sequential_monroe(scores, numwin=1)
            winners2, ties2, output = score.score(scores, numwin=1)
            assert np.all(winners1 == winners2)
            assert np.all(ties1 == ties2)
        
        
        



#    
if __name__ == '__main__':
    # unittest.main(exit=False)
    # import logging
    # logging.basicConfig()
    # logger = logging.getLogger('votesim.votemethods.score')
    # logger.setLevel(logging.DEBUG)
    
    
    t= TestSequentialMonroe()
    t.test_result()
    t.test_spatial()
    
    # t = TestRRW()
    # t.test_result()
    # t.test_result2()
    
    
    # t = TestStar()
    # # t.test_tie()
    # # t.test_tie3()
    # # t.test_wiki()
    # # t.test_zeros()
    # # t.test_tally()
    # t.test_multiwinner()
    
    # # t = TestStar()
    # # t.test_zeros()
    