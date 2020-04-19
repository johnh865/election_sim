# -*- coding: utf-8 -*-
"""
Test simple condorcet cycles
"""

import numpy as np
import votesim
from votesim.models import spatial
from votesim.votesystems.condcalcs import (
                                           pairwise_rank_matrix,
                                           has_cycle,
                                           )

pairs =[
[0, 1, 10],
[1, 2, 10],
[2, 3, 10],
[3, 1, 10]
        ]

# Sort the pairs with highest margin of victory first. 
pairs = np.array(pairs)
#c = _CycleDetector(pairs)
assert has_cycle(pairs)


votesim.logger.debug('\n\n\n.............................................')

pairs = [
[0, 1, 10],
[1, 2, 10],
[2, 3, 10],       
       ]
pairs = np.array(pairs)

assert not has_cycle(pairs)