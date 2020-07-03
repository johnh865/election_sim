"""
Test error generation that error magnitude is indeed growing
"""
# -*- coding: utf-8 -*-

import numpy as np
import sys
from votesim.models import spatial
from votesim import ballot


etype = 'score5'
vnum = 20
rs = np.random.RandomState(0)
distances = rs.rand(20, 3)
tol = np.ones((20, 3)) * 0.75
b = ballot.gen_honest_ballots(distances=distances, tol=tol,
                              base='linear')

bt = ballot.TacticalBallots(etype, b)
bt = bt.compromise()

ii = np.arange(len(bt.ratings))
jj = bt.preferred_frontrunner

assert np.all(bt.ratings[ii, jj] == 1)
assert np.all(bt.ranks[ii, jj] == 1)