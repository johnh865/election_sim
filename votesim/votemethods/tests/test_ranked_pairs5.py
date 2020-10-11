"""
Test a Ranked Pairs case where two source nodes are found in the graph.
For this test case we ought to return a tie. 
"""
# -*- coding: utf-8 -*-
import re
from ast import literal_eval

import numpy as np
import votesim
from votesim.models import spatial
from votesim.votemethods.condorcet import ranked_pairs
import votesim.votemethods.condcalcs as condcalcs

a = \
"""[[0 1 2]
 [1 0 2]
 [0 1 2]
 [0 1 2]
 [1 0 2]
 [0 1 2]
 [1 0 2]
 [0 1 2]
 [1 0 2]
 [1 0 2]]"""

a = a.replace('[ ', '[')
a = re.sub('\s+', ',', a)
a = np.array(literal_eval(a))


w,t,o = ranked_pairs(a)
w2 = condcalcs.condorcet_check_one(a)

assert len(w)==0
assert len(t)==3

#pairs = o['pairs']
#c = _CycleDetector(pairs[0:12])
#
#
#
#
#cf = c.any_circuits()




