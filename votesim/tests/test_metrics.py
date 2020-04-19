# -*- coding: utf-8 -*-

import votesim
from votesim.models import spatial
seed = None

v = spatial.SimpleVoters(seed=seed)
v.add_random(100)
c = spatial.Candidates(v, seed=seed)
c.add_random(5)
c.add_random(2, sdev=5)
e = spatial.Election(v, c, seed=seed)
e.run(etype='plurality')
e.run(etype='irv')

m = v.ElectionStats
p = e.get_parameters()


string = ''
for k, v in p.items():
    print('%40s =' % k, v)
    
print('\n\n\n')
    
for k, v in e.results.items():
    print(k, v)