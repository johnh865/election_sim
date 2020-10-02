"""
Test error generation that error magnitude is indeed growing
"""
# -*- coding: utf-8 -*-

import numpy as np
import sys
from votesim.models import spatial
from votesim import ballot


etype = 'score5'
vnum = 8
rs = np.random.RandomState(0)
distances = rs.rand(vnum, 3)
# tol = np.ones((20, 3)) * 0.75
tol = None
b = ballot.gen_honest_ballots(distances=distances, tol=tol,
                              base='linear')

ratings_honest = b.ratings


## Test compromise strategy
def test_compromise():
    bt = ballot.TacticalBallots(etype, b)
    bt = bt.compromise()
    
    ii = np.arange(len(bt.ratings))
    jj = bt.preferred_frontrunner
    
    assert np.all(bt.ratings[ii, jj] == 1)
    assert np.all(bt.ranks[ii, jj] == 1)
    
    
# def test_bury()