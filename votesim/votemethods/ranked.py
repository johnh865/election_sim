# -*- coding: utf-8 -*-
"""Classic ranked voting systems which are not IRV or Condorcet."""

import logging
import numpy as np
from votesim.votemethods.tools import (rcv_reorder,
                                       winner_check)

logger = logging.getLogger(__name__)



def borda(data, numwin=1,):
    
    vnum, cnum = data.shape    
    data1 = data.copy()
    data1[data1 == 0] = cnum
    points = cnum - data1
    tally = np.sum(points, axis=0)
    winner, ties = winner_check(tally, numwin=numwin)
    output = {}
    output['tally'] = tally
    return winner, ties, output


def bucklin(data, numwin=1,):
    vnum, cnum = data.shape
    quota = np.ceil(vnum / 2.)
    vcount = np.zeros(cnum)
    winners = []
    history = []
    for ii in range(1, cnum):
    
        vcount_ii = np.sum(data == ii, axis=0)
        vcount = vcount + vcount_ii
        history.append(vcount)
        vmax = np.max(vcount)
        if vmax >= quota:
            winner_ii, ties = winner_check(vcount, numwin=1)
            winners.extend(winner_ii)
            if len(winners) >= numwin:
                break

    winners = np.array(winners)
    output = {}
    output['tally'] = vcount
    output['history'] = np.array(history)
    return winners, ties, output
    
            
        
        
        