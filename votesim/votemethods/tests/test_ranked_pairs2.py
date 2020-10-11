"""
Test a Ranked Pairs case where two source nodes are found in the graph.
For this test case we ought to return a tie. 
"""
# -*- coding: utf-8 -*-

import numpy as np
import votesim
from votesim.models import spatial
from votesim.votemethods.condorcet import ranked_pairs


r = [[2, 0, 1, 3],
       [1, 0, 2, 3],
       [2, 0, 1, 3],
       [2, 0, 1, 3],
       [3, 2, 0, 1],
       [2, 0, 1, 3],
       [3, 2, 0, 1],
       [2, 0, 1, 3],
       [1, 0, 3, 2],
       [2, 0, 3, 1],
       [2, 0, 3, 1],
       [2, 0, 3, 1],
       [2, 0, 1, 3],
       [3, 1, 0, 2],
       [2, 0, 1, 3],
       [3, 2, 0, 1],
       [2, 0, 1, 3],
       [2, 0, 1, 3],
       [1, 0, 2, 3],
       [2, 0, 1, 3]]

r = np.array(r)

w,t,o = ranked_pairs(r)

assert len(w) == 0
assert 0 in t
assert 2 in t
assert len(t) == 2