# -*- coding: utf-8 -*-

import numpy as np
import sys
from votesim.models import spatial


v = spatial.SimpleVoters(seed=0)
v.add_points(100, 3, 1)

assert len(np.unique(v.voters, axis=0)) == 3


v = spatial.SimpleVoters(seed=0)
v.add_points(100, 3, 2)

assert len(np.unique(v.voters, axis=0)) == 3
