# -*- coding: utf-8 -*-
import numpy as np
from votesim.votemethods.ranked import borda
ranks = [[3, 2, 1],
         [0, 1, 0],
         [1, 3, 2],
         [0, 1, 0],
         [1, 3, 2],]

ranks = np.array(ranks)
w,t, o = borda(ranks)