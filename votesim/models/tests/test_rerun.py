# -*- coding: utf-8 -*-

import pdb
import numpy as np
from votesim.models import spatial


def test():
    v = spatial.Voters(0)
    v.add_random(20, 1)
    c = spatial.Candidates(v, 0)
    c.add([[0], [1], [2]])
    
    e = spatial.Election(voters=v, candidates=c)
    e.run(etype='irv')
    e.run(etype='plurality')
    
    d2o = e.dataseries(0)
    d3o = e.dataseries(1)
    
    e2 = e.rerun(d2o)
    e3 = e.rerun(d3o)
    
    d2 = e2.dataseries()
    d3 = e3.dataseries()
    
    assert np.all(e.result.ballots == e3.result.ballots)
    
    assert d2.equals(d2o)
    assert d3.equals(d3o)
    
    for key, value in d2.items():
        print(key)
        assert np.all(d2[key] == d2o[key])
        
        
def test2():
    """Check that seed correctly re-set"""
    v = spatial.Voters(seed=10)
    v.add_random(20, 1)
    c = spatial.Candidates(v, seed=5)
    c.add_random(2)
    e = spatial.Election(voters=v, candidates=c, seed=0)
    e.run(etype='plurality')
    s = e.dataseries()

    v = spatial.Voters(seed=1)
    v.add_random(5, 1)
    c = spatial.Candidates(v, seed=5)
    c.add_random(2)
    e = spatial.Election(voters=v, candidates=c, seed=0)
    er = e.rerun(s)
    assert er.voters[0].seed == 10
    assert er.voters[0].pref.__len__() == 20
    
    return

        

if __name__ == '__main__':
    test()
    test2()