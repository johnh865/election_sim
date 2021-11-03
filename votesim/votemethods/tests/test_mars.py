# -*- coding: utf-8 -*-

import pdb
import numpy as np

from votesim.votemethods.hybrid import mars


def test_electowiki():
    """https://electowiki.org/wiki/MARS_voting 
    Retrieved 2021-07-19"""
        # M N C K
    d = [[5, 2, 1, 0]]*42 + \
        [[0, 5, 2, 1]]*26 + \
        [[0, 3, 5, 3]]*15 + \
        [[0, 2, 4, 5]]*17
    d = np.array(d)
    
    w, t, o = mars(d)
    
    return



if __name__ == '__main__':
    test_electowiki()
    