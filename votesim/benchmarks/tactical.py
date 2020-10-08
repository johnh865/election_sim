# -*- coding: utf-8 -*-

import numpy as np

import votesim
import votesim.benchmarks.runtools as runtools
from votesim.models import spatial
from votesim import votesystems

strategy_ranked = {}
strategy_ranked['r1'] = ['bury']
strategy_ranked['r2'] = ['deep_bury']
strategy_ranked['r3'] = ['compromise', 'bury']
strategy_ranked['r4'] = ['compromise', 'deep_bury']
strategy_ranked['r5'] = ['truncate_hated']
strategy_ranked['r6'] = ['truncate_preferred']
strategy_ranked['r7'] = ['bullet_preferred']

strategy_scored = {}
strategy_scored['s1'] = ['bury']
strategy_scored['s2'] = ['compromise', 'bury']
strategy_scored['s3'] = ['truncate_hated']
strategy_scored['s4'] = ['truncate_preferred']
strategy_scored['s5'] = ['bullet_preferred']
strategy_scored['s6'] = ['minmax_hated']
strategy_scored['s7'] = ['minmax_preferred']

strategy_plurality = {}
strategy_plurality['p1'] = ['bullet_preferred']


def tactical_model(name, methods, 
                 seed=0,
                 numvoters=100,
                 cnum=3,
                 trialnum=1,
                 ndim=1,
                 stol=1,):
    """Tactical Election model """

    e = spatial.Election(None, None, seed=seed, name=name)

    strategy = {}
    strategy['tol'] = stol

    vh = spatial.Voters(seed=seed, strategy=strategy)
    vh.add_random(numvoters, ndim=ndim)
    vh.electionStats.set_categories([], fulloutput=True)        
    
    vt = vh.copy()
    vt.set_strategy()
    
    
    cseed = seed * trialnum
    for trial in range(trialnum):
        c = spatial.Candidates(v, seed=trial + cseed)
        c.add_random(cnum, sdev=1.5)
        e.set_models(voters=v, candidates=c)
        
        # Save parameters
        e.user_data(
                    num_voters=numvoters,
                    num_candidates=cnum,
                    num_dimensions=ndim,
                    strategy=strategy,
                    voter_tolerance=stol
                    )
        
        
        e_tactical = e.copy()
        e_tactical.set
        
        for method in methods:
            e.run(etype=method)
            
    return e