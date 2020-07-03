# -*- coding: utf-8 -*-
import unittest

import numpy as np
import votesim
import votesim.models.spatial as spatial

class TestSpatial(unittest.TestCase):
    def test1(self):
        v = spatial.Voters(0)
        voters = np.array([[-2,-1,0,1,2,]]).T
        
        v.add(voters)
        c = spatial.Candidates(v, 0)
        
        clocs = np.array([[-1, 0, 1]]).T
        c.add(clocs)
        
        e = spatial.Election(v, c)
        e.run('score')
        self.assertTrue(np.all(e.result.winners == 1))
        
                
        
            
if __name__ == '__main__':
    unittest.main()