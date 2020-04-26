# -*- coding: utf-8 -*-


import numpy as np
from votesim.models import spatial
from votesim.utilities import flatten_dict





def test():
    v = spatial.SimpleVoters(0)
    v.add_random(20, 1)
    c = spatial.Candidates(v, 0)
    c.add([[0], [1], [2]])
    
    e = spatial.Election(voters=v, candidates=c)
    e.run(etype='irv')
    e.run(etype='plurality')
    
    e2 = e.rerun(index=0)
    e3 = e.rerun(index=1)
    
    
    d2o = e.dataseries(0)
    d3o = e.dataseries(1)
    
    d2 = e2.dataseries()
    d3 = e3.dataseries()
    
    assert np.all(e.ballots == e3.ballots)
    
    for key, value in d2.items():
        print(key)
        assert np.all(d2[key] == d2o[key])
        
        
        

if __name__ == '__main__':
    import votesim
    import logging
    votesim.logconfig.start_debug()
    logger = logging.getLogger(__name__)
    
    test()