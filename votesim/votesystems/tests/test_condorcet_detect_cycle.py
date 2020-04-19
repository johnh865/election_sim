"""
Run a random construction of win-loss pairs, see if we can detect a cycle.
"""

# -*- coding: utf-8 -*-
import numpy as np
import votesim
from votesim.models import spatial
from votesim.votesystems.condorcet import (ranked_pairs,
                                           pairwise_rank_matrix,)


from votesim.votesystems.condcalcs import has_cycle, VoteMatrix


v = spatial.SimpleVoters(seed=None,)
v.add_random(200, 3)

c = spatial.Candidates(voters=v, seed=2, )
c.add_random(20)

e = spatial.Election(voters=v, candidates=c, seed=2)

ranks = e.ranks


vm = VoteMatrix(ranks=ranks)
pairs = vm.pairs
print(has_cycle(pairs))