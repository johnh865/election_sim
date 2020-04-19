# -*- coding: utf-8 -*-


import numpy as np

import votesim
from votesim.models import vcalcs, spatial
from votesim.votesystems import tools




ratings = np.random.rand(20, 4)


vnum, cnum = ratings.shape
climits = climits[:, None]


if rs is None:
    rs = np.random.RandomState()
    
p = rs.rand(vnum, cnum)
p = np.argsort(p, axis=1)
mask = p < climits


v1 = spatial.ErrorVoters(seed=0)
v1.add_random(1000, 1, error_mean=0.0, error_width=error_width)