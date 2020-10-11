# -*- coding: utf-8 -*-


import unittest
import numpy as np
import votesim

from votesim.votemethods import plurality


class TestPlurality(unittest.TestCase):
    
    
    def test_result_two(self):
        print('\ntesting 2 winners')
        a = [[0, 0, 1],
             [1, 0, 0],
             [0, 0, 1],
             [0, 1, 0],
             [1, 0, 0],
             [0, 0, 1],
             [0, 1, 0],
             [1, 0, 0],
             [0, 0, 0]]
    
        w, t, r = plurality.plurality(a, 2)
        print(w)
        self.assertTrue(0 in w)
        self.assertTrue(2 in w)
        return 
    
    
    def test_result_tie(self):
        print('\ntesting winner tie')
        a = [[0, 0, 1],
             [1, 0, 0],
             [0, 0, 1],
             [0, 1, 0],
             [1, 0, 0],
             [0, 0, 1],
             [0, 1, 0],
             [1, 0, 0],
             [0, 0, 0]]
    
        w, t, r = plurality.plurality(a, 1)
        print(w)
        self.assertTrue(0 in t)
        self.assertTrue(2 in t)
        return     
    
    
    
if __name__ == '__main__':
    unittest.main()
    