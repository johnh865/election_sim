# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 17:38:18 2020

@author: John
"""
import logging
import numpy as np
from votesim.models import spatial
from votesim.strategy.stratrunner import StrategicRunner


logging.basicConfig()
logger = logging.getLogger('votesim.strategy')
logger.setLevel(logging.WARNING)

method = 'plurality'
cnum = 5
vnum = 31
ndim = 2
name = 'strat-1'
record = None
for seed in range(10, 30):
    v = spatial.Voters(seed=seed)
    v.add_random(vnum, ndim=ndim)
    c = spatial.Candidates(v, seed=seed)
    c.add_random(cnum, sdev=1.5)
    srunner = StrategicRunner(name, 
                              method,
                              voters=v, 
                              candidates=c,
                              record=record)
    record = srunner.record
    
    print('\n')
    print('VSE honest', srunner.vse_honest)
    print('VSE 1-sided', srunner.vse_onesided)
    try:
        print('VSE change, underdog', np.max(srunner.dvse_underdogs))
    except: 
        pass
    try:
        print('VSE change, topdog', np.min(srunner.dvse_topdogs))
    except:
        pass
    print('VSE 2-sided', srunner.vse_twosided)
    print('# of viable underdogs', srunner.viable_underdogs)
    
df = srunner.record.dataframe()