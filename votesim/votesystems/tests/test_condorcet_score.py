# -*- coding: utf-8 -*-
import numpy as np

from votesim.votesystems import condcalcs, condorcet
from votesim.models import spatial
import votesim
seed = None


for i in range(10):
    v = spatial.Voters(seed=seed)
    v.add_random(20, 3)
    c = spatial.Candidates(v, seed=seed)
    c.add_random(4)
    
    e = spatial.Election(v, c, scoremax=5)
    b = e.vballots.honest_ballots
    # e.result.
    
    ratings = b.ratings
    
    
    p1 = condcalcs.pairwise_scored_matrix(b.ratings)
    p2 = condcalcs.pairwise_rank_matrix(b.ranks)
    p3 = condcalcs.pairwise_scored_matrix(b.scores)
    
    w1, *args = condcalcs.condorcet_winners_check(matrix=p1)
    w2, *args = condcalcs.condorcet_winners_check(matrix=p2)
    w3, *args = condcalcs.condorcet_winners_check(matrix=p3)
    
    
    s1 = condcalcs.smith_set(vm=p1)
    s2 = condcalcs.smith_set(vm=p2)
    s3 = condcalcs.smith_set(vm=p3)
    
    print('')
    print(w1, w2, w3)
    print(s1, s2, s3)
    
    
    w,t,o = condorcet.smith_score(b.scores)
    print('winner', w)
    print(o)
    
    assert np.all(p1 == p2)
    if len(w1) > 0:
        assert w1 == w2

