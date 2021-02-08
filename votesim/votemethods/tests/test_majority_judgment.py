# -*- coding: utf-8 -*-
import numpy as np
import votesim

# votesim.logSettings.start_debug()
from votesim.votemethods.score import majority_judgment
from votesim.models import spatial

def test_run():
    for seed in range(50):
        v = spatial.Voters(seed=seed)
        v.add_random(20)
        c = spatial.Candidates(v, seed=seed)
        c.add_random(6)
        e = spatial.Election(voters=v, candidates=c, seed=0,)
        e.run('maj_judge')
        
        # scores = e.output[0]['round_history']
        scores = e.result.runner.output['round_history']
        
        
        print('ratings for each elimination round')
        print(scores)
        print('winner=%s' % e.result.winners)
        print('')



def test_case():
    """Test a case that failed during simple test benchmark.
    After investigation it seems like this is a case where
    all ballot scores are zero.
    """
    seed = 0
    numvoters = 101
    cnum = 3
    trial = 54
    trialnum = 100
    ndim = 2
    stol = 0.25
    base = 'linear'
    name = 'test'
    
    e = spatial.Election(None, None, seed=seed, name=name)

    v = spatial.Voters(seed=seed, tol=stol, base=base)
    v.add_random(numvoters, ndim=ndim)
    cseed = seed * trialnum

    c = spatial.Candidates(v, seed=trial + cseed)
    c.add_random(cnum, sdev=1.5)
    e.set_models(voters=v, candidates=c)
    
    ballots = e.ballotgen.get_honest_ballots('maj_judge')
    result = e.run('maj_judge')
    assert np.all(result.ties == [0, 1, 2])
    return

    
    
if __name__ == '__main__':
    test_case()
    test_run()