# -*- coding: utf-8 -*-


import votesim
from votesim.models import spatial
from votesim.votemethods.condorcet import ranked_pairs
v = spatial.Voters(seed=0,)
v.add_random(50)
seed = None
c = spatial.Candidates(voters=v, seed=seed, )
c.add_random(12)

e = spatial.Election(voters=v, candidates=c, seed=2)
# e.run(method=ranked_pairs, btype='rank')
e.run(etype='ranked_pairs')
