# -*- coding: utf-8 -*-

import unittest
import logging

import numpy as np
from votesim.votesystems import irv
import votesim

logger = logging.getLogger(__name__)


class TestIRV(unittest.TestCase):
    
    def test_tie(self):
        print('TEST TIE #1')
        d = [[1, 2,],
             [2, 1,]]
        
        winners1, ties1, h = irv.irv_stv(d, 1)
        winners2, ties2, output = irv.irv(d, 1)
        print('winners1', winners1)
        print('winners2', winners2)
        print('ties1', ties1)
        print('ties2', ties2)        
        self.assertTrue(len(winners1) == 0)
        self.assertTrue(len(winners2) == 0)
        self.assertTrue(
                 np.all(np.in1d(ties1, ties2))
                 )


        
        self.assertEqual(len(winners1), 0)
        self.assertEqual(len(ties1), 2)
        self.assertTrue(0 in ties1)
        self.assertTrue(1 in ties1)
        winners2, ties2, o = irv.irv(d, 1)
        return 
    
    
    def test_tie2(self):
        print('TEST TIE #2')
        d = [[1,2,3],
             [1,3,2]]
        
#        winners, ties, h = irv.IRV_STV(d, 2)
        winners, ties, h = irv.irv_stv(d, 2)
        print('winners', winners)
        print('ties', ties)
        
        self.assertTrue(0 in winners)
        return
        
        
    def test_eliminate(self):
        d = [[1, 2, 3, 4],
             [1, 3, 2, 4],
             [3, 2, 1, 4],
             [2, 3, 1, 4],
             [3, 0, 2, 1]]        
        d = np.array(d)
        
        first_round = [
             [1, 0, 2, 3],
             [1, 0, 2, 3],
             [2, 0, 1, 3],
             [2, 0, 1, 3],
             [3, 0, 2, 1],
             ]
        
        second_round = [
                [1, 0, 2, 0],
                [1, 0, 2, 0],
                [2, 0, 1, 0],
                [2, 0, 1, 0],
                [2, 0, 1, 0]]
        
        third_round = [
                [0, 0, 1, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 0]]         
        
        first_round = np.array(first_round)
        second_round = np.array(second_round)
        logger.info('start votes\n%s', d)
        logger.info(d)
        
        
        d1, loser, ties, h = irv.irv_eliminate(d)
        logger.info('1st round results\n%s', d1)
        self.assertTrue(np.all(first_round == d1))
        
        
        d2, loser, ties, h = irv.irv_eliminate(d1)
        logger.info('2nd round results\n%s', d2)
        self.assertTrue(np.all(second_round == d2))
        
        d3, loser, ties, h = irv.irv_eliminate(d2)
        logger.info('3rd round results\n%s', d3)
        self.assertTrue(np.all(third_round == d3))
        
        w, t, h = irv.irv_stv(d, numwin=1)
        self.assertIn(2, w)
        return
    
    
    def test_stv(self):
        print('TEST STV')
        d = [[1, 2, 3, 4],
             [1, 3, 2, 4],
             [3, 2, 1, 4],
             [2, 3, 1, 4],
             [3, 0, 2, 1]]
        d = np.array(d)
        winners, ties, h = irv.irv_stv(d, 2)
        self.assertTrue(0 in winners)
        self.assertTrue(2 in winners)
        return
    
    
    def test_RCVReorder(self):
        print('\nTEST RCV ReOrder')
        
        a = [[1, 5, 2, 0, 4, 10],
             [2, 3, 4, 5, 6, 7],
             [0, 0, 0, 5, 6, 7]]
        a = np.array(a)
        b = irv.rcv_reorder(a)
        
        
        correct = [
                [1, 4, 2, 0, 3, 5],
                [1, 2, 3, 4, 5, 6],
                [0, 0, 0, 1, 2, 3]
                ]
        correct = np.array(correct)
        
        compare = np.all(correct == b)
        self.assertTrue(compare)
        return
        
        
    def test_wiki(self):
        """
        Test example from wikipedia, retrieved Dec 19, 2019.
        Correct results taken from wikipedia (winner knoxville K)
        https://en.wikipedia.org/wiki/Instant-runoff_voting
        """
        
        # M N C K
        d = [[1, 2, 3, 4]]*42 + \
            [[4, 1, 2, 3]]*26 + \
            [[4, 3, 1, 2]]*15 + \
            [[4, 3, 2, 1]]*17
        d = np.array(d)
        winners, ties, history = irv.irv_stv(d, 1)
#        print('test wiki')
#        print('winners=\n', winners)
#        print('history=\n', history)
#        
        correct_history = [[42, 26, 15, 17],
                           [42, 26,  0, 32],
                           [42,  0,  0, 58]]
        
        correct_history = np.array(correct_history)
            
        self.assertTrue(np.all(correct_history == history))
        self.assertEqual(winners[0], 3)
        
        
    def test_irv2(self):
        
        
        success_count = 0
        fail_count = 0
#        print('test_irv2 -- compared STV vs IRV')
        rstate = np.random.RandomState()
        for seed in range(60):
            rstate.seed(seed)
            ratings = rstate.rand(100, 5)
            ranks = votesim.votesystems.tools.score2rank(ratings)
#            print(seed)
            w1, t1, o1 = irv.irv_stv(ranks)            
            w2, t2, o2 = irv.irv(ranks)
     
            w1 = np.sort(w1)
            w2 = np.sort(w2)
            t1 = np.sort(t1)
            t2 = np.sort(t2)
            
#            print('Seed # %s' % seed)
                  
            success = np.all(w1 == w2) & np.all(t1 == t2)
#            print('Methods same result?', success)
            if success:
                success_count += 1
            else:
                fail_count += 1
#            
#                print('FAILED METHOD IRV INPUT')
#                print(ranks)
#                print('\n\nRUNNING STV RESULTS')
#
#                print('\n\nRUNNING IRV RESULTS')
#
#                print('history')
#                print(o1)
#                print(o2['round_history'])
    
#            print('winners=%s', w1)
#            print('ties=%s', t1)        
#            print('winners=%s', w2)
#            print('ties=%s', t2)                   
#                
#        print('# of successes =', success_count)
#        print('# of fails =', fail_count)
        self.assertTrue(fail_count == 0)  
        return
          
        
    def test_irv_tie3(self):
        d =  [[5,2,1,4,3],
             [3,5,2,1,4],
             [2,3,1,5,4],
             [2,3,5,1,4],
             [5,4,1,3,2],
             [3,2,5,4,1],
             [1,4,3,5,2],
             [5,2,3,1,4],
             [3,5,4,2,1],
             [1,4,3,2,5],
             ]
        d = np.array(d)
        w2, t2, o2 = irv.irv(d)
      

    def test_stv_tie3(self):
        d =  [[5,2,1,4,3],
             [3,5,2,1,4],
             [2,3,1,5,4],
             [2,3,5,1,4],
             [5,4,1,3,2],
             [3,2,5,4,1],
             [1,4,3,5,2],
             [5,2,3,1,4],
             [3,5,4,2,1],
             [1,4,3,2,5],
             ]
        d = np.array(d)
        w2, t2, o2 = irv.irv_stv(d)
        
        

    def test_stv_tie4(self):
        d =  [[2,4,3,1,5]
            ,[5,2,1,4,3]
            ,[1,3,4,5,2]
            ,[1,3,5,2,4]
            ,[3,1,2,4,5]
            ,[2,5,3,1,4]
            ,[2,1,4,3,5]
            ,[3,1,5,2,4]
            ,[1,2,3,5,4]
            ,[3,2,5,4,1]]

        d = np.array(d)
        w2, t2, o2 = irv.irv_stv(d)        
        
        
    
if __name__ == '__main__':
    pass

    # logging.basicConfig()
    # logger = logging.getLogger('votesim.votesystems.irv')
    # logger.setLevel(logging.DEBUG)
    
    # t = TestIRV()
    # t.test_tie()
    unittest.main(exit=False)
    # a = TestIRV()
    # a.test_eliminate()
    # a.test_irv2()
    # a.test_wiki()
    # a.test_stv_tie4()
    
    
    
    