# -*- coding: utf-8 -*-
from votesim.votemethods import irv
import numpy as np

def test_stv1():

        # M N C K
    d = [[1, 2, 3, 4]]*43 + \
        [[4, 1, 2, 3]]*26 + \
        [[4, 3, 1, 2]]*15 + \
        [[4, 3, 2, 1]]*17 + \
        [[1, 2, 3, 0]]
        
    d = np.array(d)
    
    for ii in range(100):
        winners1, ties1, output = irv.irv_stv(d, 3, seed=ii)
        winners2, ties2, output = irv.irv_stv(d, 3, reallocation='gregory')
        assert np.all(winners1[0:2] == winners2[0:2])
    return

    
    
if __name__ == '__main__':
    test_stv1()