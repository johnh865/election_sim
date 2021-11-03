"""Tactical voting benchmarks that do not assume an underdog.
Tactical_v3 includes 2 factions for a a bimodal voter distribution."""

import numpy as np
import votesim.benchmarks.runtools as runtools
from votesim.models import spatial
from votesim.metrics import TacticCompare

from votesim.benchmarks.tactical_v2 import get_tactics


def tactical_model_v3(
        name: str, 
        methods : list, 
        seed=0,
        numvoters=51,
        cnum=5,
        cstd=1.5,
        ndim=1,
        tol=None,
        numfactions=2,
        ratio=1.0,
        ) -> spatial.Election:
    """Tactical Election model that test every single candidate as an underdog,
    and tests topdog resistance using bullet voting. 
    """

    e = spatial.Election(None, None, seed=seed, name=name)

    # Construct base strategy
    strategy_base = {}
    strategy_base['ratio'] = ratio
    strategy_base['subset'] = 'underdog'
    
    # Create underdog strategy 
    strategy2 = strategy_base.copy()
    
    # Create topdog strategy
    strategy3 = strategy_base.copy()
    strategy3['tactics'] = ['bullet_preferred']    
    strategy3['subset'] = 'topdog'
    
    # Generate voters
    
    # Generate voter centroid locations
    rs = np.random.RandomState(seed=[seed, 1000])
    loc = rs.uniform(low=-2, high=2, size=(numfactions, ndim)) 
    scale = rs.uniform(low=.5, high=1.5, size=(numfactions))
    numvoters_list = rs.randint(low=int(numvoters/4), high=numvoters, size=numfactions)
    
    v = spatial.Voters(seed=seed, tol=tol, base='linear')
    for ii in range(numfactions):
        v.add_random(numvoters_list[ii],
                     ndim=ndim, 
                     loc=loc[ii], 
                     scale=scale[ii])
    
    # Generate candidates 
    c = spatial.Candidates(v, seed=seed)
    c.add_random(cnum, sdev=cstd)    
    e.set_models(voters=v, candidates=c)
    
    # Construct election identification
    eid = (seed, numvoters, cnum, ndim,)
    
    for method in methods:
        
        # Set empty (honest) strategy
        e.set_models(strategies=spatial.StrategiesEmpty())
        e.user_data(eid=eid, strategy='honest')            
        result1 = e.run(etype=method)
        winner = result1.winners[0]
        stats_honest = result1.stats
        
        underdog_list = list(range(cnum))
        underdog_list.remove(winner)
    
        # test each underdog    
        for underdog in underdog_list:
            strategy2['underdog'] = underdog       
            strategy3['underdog'] = underdog
                
            # test each tactic
            tactics = get_tactics(method)
            for tactic in tactics:
                strategy2['tactics'] = tactic
                        
                # Run one-sided strategy
                s = spatial.Strategies(v).add(**strategy2)
                e.set_models(strategies=s)
                e.user_data(eid=eid, strategy='one-sided')            
                result2 = e.run(etype=method, result=result1)
                
                # Create tactical comparison output, add to output
                tactic_compare = TacticCompare(
                    e_strat=result2.stats,
                    e_honest=stats_honest)
                e.append_stat(tactic_compare)
                
                # Run two-sided strategy with top-dog bullet vote.
                s.add(**strategy3)
                e.set_models(strategies=s)
                e.user_data(eid=eid, strategy='two-sided')            
                result3 = e.run(etype=method, result=result1)
                
                # Create tactical comparison output, add to output
                tactic_compare = TacticCompare(
                    e_strat=result3.stats,
                    e_honest=stats_honest)
                e.append_stat(tactic_compare)
            
    return e



def tactical_v3_0():
    
    name = 'tactical_v3_0'
    model = tactical_model_v3
    kwargs = {}    
    kwargs['name'] = name
    kwargs['seed'] = np.arange(100)
    kwargs['numvoters'] = 51
    kwargs['ndim'] = [1, 2,]
    kwargs['cnum'] = [5]
    
    case_args = runtools.CaseGenerator(**kwargs)
    benchmark = runtools.CreateBenchmark(name, model, case_args)    
    return benchmark




def tactical_v3_1():
    
    name = 'tactical_v3_1'
    model = tactical_model_v3
    kwargs = {}    
    kwargs['name'] = name
    kwargs['seed'] = np.arange(1000)
    kwargs['numvoters'] = 51
    kwargs['ndim'] = [1, 2, 3]
    kwargs['cnum'] = [3, 5]
    
    case_args = runtools.CaseGenerator(**kwargs)
    benchmark = runtools.CreateBenchmark(name, model, case_args)    
    return benchmark