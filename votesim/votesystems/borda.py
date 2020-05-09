# -*- coding: utf-8 -*-

import numpy as np
import logging
from votesim.votesystems.tools import (RCV_reorder,
                                       droop_quota, 
                                       winner_check)

logger = logging.getLogger(__name__)



def borda(data, numwin=1,):
    
    vnum, cnum = data.shape    
    points = cnum - data
    winner, ties = winner_check(points, numwin=numwin)
    return winner, ties, points
