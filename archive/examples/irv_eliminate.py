# -*- coding: utf-8 -*-

import votesim
from votesim.votesystems.irv import irv_eliminate
import numpy as np

ratings = np.random.rand(6, 5)
rankings = votesim.behavior.score2rank(ratings, cutoff=.1)
d = rankings

print(d, '\n')
out = irv_eliminate(d)
d = out[0]
loser = out[1]
ties = out[2]
print(d, '\n')
print('loser=%s, ties=%s' % (loser, ties))


print(d, '\n')
out = irv_eliminate(d)
d = out[0]
loser = out[1]
ties = out[2]
print(d, '\n')
print('loser=%s, ties=%s' % (loser, ties))


print(d, '\n')
out = irv_eliminate(d)
d = out[0]
loser = out[1]
ties = out[2]
print(d, '\n')
print('loser=%s, ties=%s' % (loser, ties))


print(d, '\n')
out = irv_eliminate(d)
d = out[0]
loser = out[1]
ties = out[2]
print(d, '\n')
print('loser=%s, ties=%s' % (loser, ties))