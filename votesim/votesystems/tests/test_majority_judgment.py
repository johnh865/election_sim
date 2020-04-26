# -*- coding: utf-8 -*-
import votesim

votesim.logSettings.start_debug()
from votesim.votesystems.score import majority_judgment



from votesim.models import spatial




for seed in range(300):
    v = spatial.SimpleVoters(seed=seed)
    v.add_random(20)
    c = spatial.Candidates(v, seed=seed)
    c.add_random(6)
    e = spatial.Election(voters=v, candidates=c, seed=0,)
    e.run('majority_judgment')
    
    scores = e.output[0]
    
    print('ratings for each elimination round')
    print(scores)
    print('winner=%s' % e.winners)
    print('')
