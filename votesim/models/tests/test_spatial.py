# -*- coding: utf-8 -*-

import unittest
import numpy as np
import pandas as pd
import sys
from votesim.models import spatial





class TestSpatial(unittest.TestCase):
    def test1(self):
        """Test whether cnum limit is working"""
        
        v = spatial.Voters(0)
        v.add_random(20, 3)
        
        
        c = spatial.Candidates(v, 0)
        c.add_random(3)
        print(c.pref)
        
        e = spatial.Election(voters=v, candidates=c)
        e.user_data(a=0, b=1)
        
        print('running plurality')
        e.run(etype='plurality')
        print('running irv')
        
        e.run(etype='irv')
        e.run(etype='score')
        
        # test dataframe construction
        df = e.dataframe()
        
        assert 'args.user.a' in df.keys()
        assert 'args.user.b' in df.keys()
        return df
        
        
    def test_rerunner(self):
        
        v = spatial.Voters(0)
        v.add_random(20, 1)
        c = spatial.Candidates(v, 0)
        c.add([[0], [1], [2]])
        
        e = spatial.Election(voters=v, candidates=c)
        e.run(etype='irv')
        e.run(etype='plurality')
        
        series = e.dataseries()
        e2 = e.rerun(series)
        series2 = e.dataseries()
        
        assert series2.equals(series)
        
            
if __name__ == '__main__':
    unittest.main()
    
    
    t = TestSpatial()
    df = t.test1()
