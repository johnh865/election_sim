import numpy as np
import votesim
import votesim.benchmarks.runtools as runtools
from votesim.models import spatial_v2
from votesim import votemethods
from votesim.metrics import TacticCompare


tactics_ranked = {}
tactics_ranked['r0'] = ['bury']
tactics_ranked['r1'] = ['deep_bury']
tactics_ranked['r2'] = ['compromise', 'bury']
tactics_ranked['r3'] = ['compromise', 'deep_bury']
tactics_ranked['r4'] = ['compromise']
tactics_ranked['r5'] = ['truncate_hated']
tactics_ranked['r6'] = ['truncate_preferred']
tactics_ranked['r7'] = ['bullet_preferred']

tactics_scored = {}
tactics_scored['s0'] = ['bury']
tactics_scored['s1'] = ['compromise', 'bury']
tactics_scored['s2'] = ['compromise']
tactics_scored['s3'] = ['truncate_hated']
tactics_scored['s4'] = ['truncate_preferred']
tactics_scored['s5'] = ['bullet_preferred']
tactics_scored['s6'] = ['minmax_hated']
tactics_scored['s7'] = ['minmax_preferred']

tactics_plurality = {}
tactics_plurality['p1'] = ['bullet_preferred']


statnames = [
    'output.tactic_compare.regret_efficiency_candidate.topdog-0',
    'output.tactic_compare.regret_efficiency_candidate.underdog-0',
    
    ]



def get_tactics(etype: str) -> list:
    ballot_type = votemethods.get_ballot_type(etype)    
    
    if ballot_type == 'rank':
        return list(tactics_ranked.values())
    
    elif ballot_type == 'score' or ballot_type == 'rate':
        return list(tactics_scored.values())    
    
    elif ballot_type == 'vote':
        return list(tactics_plurality.values())
    
    
def get_topdog_tactic(etype: str) -> str:
    """Return topdog defensive strategy"""
    return 'bullet_preferred'



def tactical_model_v2(
        name: str, 
        methods : list, 
        seed=0,
        numvoters=51,
        cnum=5,
        cstd=1.5,
        ndim=1,
        tol=None,
        ratio=1.0,
        ) -> spatial_v2.Election:
    """Tactical Election model that test every single candidate as an underdog,
    and tests topdog resistance using bullet voting. 
    """

    e = spatial_v2.Election(None, None, seed=seed, name=name)

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
    v = spatial_v2.Voters(seed=seed, tol=tol, base='linear')
    v.add_random(numvoters, ndim=ndim)

    # Generate candidates 
    c = spatial_v2.Candidates(v, seed=seed)
    c.add_random(cnum, sdev=cstd)    
    e.set_models(voters=v, candidates=c)
    
    # Construct election identification
    eid = (seed, numvoters, cnum, ndim,)
    
    for method in methods:
        
        # Set empty (honest) strategy
        e.set_models(strategies=spatial_v2.StrategiesEmpty())
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
                s = spatial_v2.Strategies(v).add(**strategy2)
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



def tactical_v2_0():
    
    name = 'tactical_v2_0'
    model = tactical_model_v2
    kwargs = {}    
    kwargs['name'] = name
    kwargs['seed'] = np.arange(100)
    kwargs['numvoters'] = 51
    kwargs['ndim'] = [1, 2,]
    kwargs['cnum'] = [5]
    
    case_args = runtools.CaseGenerator(**kwargs)
    benchmark = runtools.CreateBenchmark(name, model, case_args)    
    return benchmark




def tactical_v2_1():
    
    name = 'tactical_v2_1'
    model = tactical_model_v2
    kwargs = {}    
    kwargs['name'] = name
    kwargs['seed'] = np.arange(1000)
    kwargs['numvoters'] = 51
    kwargs['ndim'] = [1, 2, 3]
    kwargs['cnum'] = [3, 5]
    
    case_args = runtools.CaseGenerator(**kwargs)
    benchmark = runtools.CreateBenchmark(name, model, case_args)    
    return benchmark