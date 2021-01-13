# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 20:24:13 2020

@author: John
"""
import numpy as np

import votesim
import votesim.benchmarks.runtools as runtools
from votesim.models import spatial
from votesim import votemethods
from votesim.metrics import TacticCompare

statnames = [
    'output.tactic_compare.regret_efficiency_candidate.tactical-topdog-0',
    'output.tactic_compare.regret_efficiency_candidate.tactical-underdog-0',
    
    ]

def tactical_model(name: str, 
                   method : str, 
                   seed=0,
                   numvoters=51,
                   cnum=5,
                   ndim=1,
                   tol=None,
                   ratio=1.0,
                   frontrunnernum=2,
                   ) -> spatial.Election:
    """Tactical Election model """

    e = spatial.Election(None, None, seed=seed, name=name)

    # Construct base strategy
    strategy_base = {}
    strategy_base['ratio'] = ratio
    strategy_base['frontrunnernum'] = frontrunnernum
    strategy_base['frontrunnertype'] = 'eliminate'
    # strategy_base['tactics'] = ('bury', 'compromise')
    strategy_base['tactics'] = ('minmax_preferred')
    strategy_base['subset'] = 'underdog'
    
    strategy2 = strategy_base.copy()
    
    # Generate voters
    v = spatial.Voters(seed=seed, tol=tol, base='linear')
    v.add_random(numvoters, ndim=ndim)

    # Generate candidates 
    c = spatial.Candidates(v, seed=seed)
    c.add_random(cnum, sdev=1.0)    
    e.set_models(voters=v, candidates=c)


    # Set empty (honest) strategy
    e.set_models(strategies=())
    result1 = e.run(etype=method)
    winner = result1.winners[0]
    stats_honest = result1.stats
    
    underdog_list = list(range(cnum))
    underdog_list.remove(winner)
    
    try:
        print('honest tally')
        print(result1.runner.output['tally'])
        print('')
    except KeyError:
        pass
    # test each underdog
    
    
    
    for underdog in underdog_list:
        # Run one-sided strategy
        strategy2['underdog'] = underdog       
        s = spatial.Strategies(v).add(strategy2, 0)
        e.set_models(strategies=s)
        
        result2 = e.run(etype=method, result=result1)
        # Create tactical comparison output, add to output
        tactic_compare = TacticCompare(e_strat=result2.stats,
                                       e_honest=stats_honest)
        e.append_stat(tactic_compare)
        series = e.dataseries()
        out1 = series[statnames[0]]
        out2 = series[statnames[1]]
        print('Setting underdog = %s' % underdog)
        print('Topdog VSE = %.2f' % out1)
        print('Underdog VSE = %.2f' % out2)
        print('winner=%s' % result2.winners[0])
        try:
            print('tally=', result2.runner.output['tally'])
        except KeyError:
            pass
        # print(result2.ballots)
        print('')
        
        
    # Calculate underdog using tally
    s0 = spatial.Strategies(v).add(strategy_base, 0)
    e.set_models(voters=v, candidates=c, strategies=s0)
    e.run(etype=method)
    frunners = e.ballotgen.tacticalballots.root.get_group_frontrunners(s0.data)
    print('calculated front runners (tally) = ', frunners)
    

       
    #Calculate underdog using eliminate
    strategy_base['frontrunnertype'] = 'eliminate'
    s0 = spatial.Strategies(v).add(strategy_base, 0)
    e.set_models(voters=v, candidates=c, strategies=s0)
    e.run(etype=method)
    frunners = e.ballotgen.tacticalballots.root.get_group_frontrunners(s0.data)
    print('calculated front runners (eliminate) = ', frunners)
    return e



e = tactical_model('test', 'approval50', seed=None)
df = e.dataframe()








