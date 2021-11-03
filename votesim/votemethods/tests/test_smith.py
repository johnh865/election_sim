# -*- coding: utf-8 -*-
import numpy as np
import pdb


from votesim.votemethods.condcalcs import smith_set

def test_smith():
    """Test classic 3 way tie"""
    ranks = [[1,3,2],
             [2,1,3],
             [3,2,1]]
    
    ranks = np.array(ranks)
    out = smith_set(ranks=ranks)
    assert 0 in out
    assert 1 in out
    assert 2 in out
    assert len(out) == 3
    return


def test_smith2():
    ranks = (
        [[1,2,3]]*7 +
        [[2,1,3]]*7 + 
        [[3,2,1]]*2 +
        [[2,3,1]]*2
        )
    ranks = np.array(ranks)
    out = smith_set(ranks=ranks)
    assert 0 in out
    assert 1 in out
    assert len(out) == 2
    return


if __name__ == '__main__':
    test_smith()
    test_smith2()