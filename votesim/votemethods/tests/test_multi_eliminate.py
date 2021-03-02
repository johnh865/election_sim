# -*- coding: utf-8 -*-
from votesim.models import spatial
from votesim.votemethods import irv, tools, score
import numpy as np

seed = 1
v = spatial.Voters(seed=seed)
v.add_random(40, 3)
c = spatial.Candidates(v, seed=seed)
c.add_random(10)

e = spatial.Election(v, c, scoremax=5)
ranks = e.ballotgen.honest_ballot_dict['rank']
scores = e.ballotgen.honest_ballot_dict['score']



def test_multi_win_score():
    
    w1, t1, o1 = score.score(scores, numwin=3)
    w2, t2, o2 = tools.multi_win_eliminate(score.score, scores, numwin=3)
    
    assert np.all(w1 == w2)
    assert np.all(t1 == t2)
    return

    
    

  
    
if __name__ == '__main__':
    test_multi_win_score()