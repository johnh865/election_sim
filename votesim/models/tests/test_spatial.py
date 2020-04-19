# -*- coding: utf-8 -*-

import unittest
import numpy as np
import sys
from votesim.models import spatial





class TestSpatial(unittest.TestCase):
    def test1(self):
        """Test whether cnum limit is working"""
        
        v = spatial.SimpleVoters(0)
        v.add_random(20, 3)
        
        
        c = spatial.Candidates(v, 0)
        c.add_random(3)
        print(c.candidates)
        
        e = spatial.Election(voters=v, candidates=c)
        print('running plurality')
        e.run(etype='plurality')
        print('running irv')
        e.run(etype='irv')
        e.run(etype='score')
        
        # test dataframe construction
        e.dataframe()
        
        
    def test_rerunner(self):
        
        v = spatial.SimpleVoters(0)
        v.add_random(20, 1)
        c = spatial.Candidates(v, 0)
        c.add([[0], [1], [2]])
        
        e = spatial.Election(voters=v, candidates=c)
        e.run(etype='irv')
        e.run(etype='plurality')
        
        e2 = e.rerun(index=0)

    
        
            
if __name__ == '__main__':
    unittest.main()