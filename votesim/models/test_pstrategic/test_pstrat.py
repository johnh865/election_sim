# -*- coding: utf-8 -*-

from votesim.models import pstrategic

v = pstrategic.SimpleVoters(seed=0)
v.add_points(50, 4, ndim=1)


cseed = None

c = pstrategic.Candidates(v, seed=cseed)
c.add_random(12, sdev=.8)

e = pstrategic.Election(voters=v, candidates=c, seed=0,)
se = pstrategic.StrategicElection(e, etype='plurality', seed=0,)
#d = se.run_iter(0, 1)

#print(d)