# -*- coding: utf-8 -*-

import numpy as np
import sys
from votesim.models import spatial


v = spatial.Voters(seed=0)
v.add_points(100, 3, 1)

assert len(np.unique(v.pref, axis=0)) == 3


v = spatial.Voters(seed=0)
v.add_points(100, 3, 2)

assert len(np.unique(v.pref, axis=0)) == 3
