# -*- coding: utf-8 -*-


import numpy as np
from votesim.models import spatial


def test():
    v = spatial.SimpleVoters(0)
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
    
    assert np.all(e.ballots == e3.ballots)
    
    assert d2.equals(d2o)
    assert d3.equals(d3o)
    
    for key, value in d2.items():
        print(key)
        assert np.all(d2[key] == d2o[key])
        
        
        

if __name__ == '__main__':
    import votesim
    import logging
    votesim.logSettings.start_debug()
    logger = logging.getLogger(__name__)
    
    test()