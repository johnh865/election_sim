# -*- coding: utf-8 -*-
import pdb
import numpy as np

import votesim
import votesim.benchmarks.runtools as runtools
from votesim.models import spatial
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


def get_strategies(etype: str) -> list:
    """Build election strategies given an votemethod name.
    
    Parameters
    ----------
    etype : str
        Voting method
    
    Returns
    -------
    out : list[dict]
        A list of available strategies for the method. 
        
    """
    keywords = votemethods.method_keywords[etype]
    ballot_type = votemethods.ballot_type(etype)    
    strategies = []
    
    if ballot_type == 'rank':
        if 'condorcet' in keywords:
            frontrunnertype = 'condorcet'
        else:
            frontrunnertype = 'tally'
        
        for tactic_list in tactics_ranked.values():
            
            
            
            
            strategy = {
                'tactics' : tactic_list,
                'frontrunnertype' : 'tally',
                'tol' : None, 
                'base' : 'linear',}
            strategies.append(strategy)
            
            strategy = {
                'tactics' : tactic_list,
                'frontrunnertype' : 'condorcet',
                'tol' : None, 
                'base' : 'linear',}
            strategies.append(strategy)
            
    elif ballot_type == 'score':
        for tactic_list in tactics_scored.values():
            
            strategy = {
                'tactics' : tactic_list,
                'frontrunnertype' : 'tally',
                'tol' : None, 
                'base' : 'linear',}
            strategies.append(strategy)        
            
    elif ballot_type == 'vote':
        for tactic_list in tactics_plurality.values():
            
            strategy = {
                'tactics' : tactic_list,
                'frontrunnertype' : 'tally',
                'tol' : None, 
                'base' : 'linear',}    
            strategies.append(strategy)        
    return strategies

  
def tactical_model(name: str, 
                   methods : list, 
                   seed=0,
                   numvoters=100,
                   cnum=3,
                   ndim=1,
                   tol=None,
                   ratio=1.0,
                   frontrunnernum=2,
                   ) -> spatial.Election:
    """Tactical Election model """

    e = spatial.Election(None, None, seed=seed, name=name)

    # Construct base strategy
    strategy_base = {}
    strategy_base['tol'] = tol
    strategy_base['ratio'] = ratio
    strategy_base['frontrunnernum'] = frontrunnernum
    
    # Generate voters
    v = spatial.Voters(seed=seed, strategy=strategy_base)
    v.add_random(numvoters, ndim=ndim)
    v.electionStats.set_categories([], fulloutput=True)      

    # Generate candidates 
    c = spatial.Candidates(v, seed=seed)
    c.add_random(cnum, sdev=2.0)
    e.set_models(voters=v, candidates=c)
    
    # Construct election identification
    eid = (seed, numvoters, cnum, ndim,)
    
    # Loop through election methods
    for method in methods:
        
        # First run the honest election
        v.set_strategy()
        e.user_data(
                    eid=eid,
                    num_voters=numvoters,
                    num_candidates=cnum,
                    num_dimensions=ndim,
                    strat_id=-1,
                    onesided=False,
                    )          
        
        e.run(etype=method)
        result = e.result.copy()
        honest_ballots = e.ballotgen.honest_ballots
        stats_honest = e.electionStats.copy()
        
        # Initialize strategy elections
        strategies = get_strategies(method)
        for s in strategies:
            s.update(strategy_base)
        
        # Iterate through available strategies
        for ii, strategy in enumerate(strategies):
            
            # Run full strategy
            onesided = False
            strategy['subset'] = ''
            v.set_strategy(**strategy)
            e.user_data(
                        eid=eid,
                        num_voters=numvoters,
                        num_candidates=cnum,
                        num_dimensions=ndim,
                        strat_id=ii,
                        onesided=onesided,
                        )            
            e.run(etype=method, result=result, ballots=honest_ballots)
            # Create tactical comparison output, add to output
            tactic_compare = TacticCompare(e_strat=e.electionStats,
                                           e_honest=stats_honest)
            e.append_stat(tactic_compare)
            
            # Run one-sided strategy
            onesided = True
            strategy['subset'] = 'underdog'
            v.set_strategy(**strategy)
            e.user_data(
                        eid=eid,
                        num_voters=numvoters,
                        num_candidates=cnum,
                        num_dimensions=ndim,
                        strat_id=ii,
                        onesided=onesided,
                        )            
            e.run(etype=method, result=result, ballots=honest_ballots)            
            tactic_compare = TacticCompare(e_strat=e.electionStats,
                                           e_honest=stats_honest)
            e.append_stat(tactic_compare)

    return e




def tactical0():
    
    name = 'tactical0'
    model = tactical_model
    kwargs = {}    
    kwargs['name'] = name
    kwargs['seed'] = np.arange(100)
    kwargs['numvoters'] = 51
    kwargs['ndim'] = [1, 2,]
    kwargs['cnum'] = [5]
    
    case_args = runtools.CaseGenerator(**kwargs)
    benchmark = runtools.CreateBenchmark(name, model, case_args)    
    return benchmark



def tactical1():
    
    name = 'tactical1'
    model = tactical_model
    kwargs = {}    
    kwargs['name'] = name
    kwargs['seed'] = np.arange(10000)
    kwargs['numvoters'] = 100
    kwargs['ndim'] = [1, 2, 3]
    kwargs['cnum'] = [3, 4, 5, 7, 9]
    
    case_args = runtools.CaseGenerator(**kwargs)
    benchmark = runtools.CreateBenchmark(name, model, case_args)    
    return benchmark




def tactical_dummy():
    
    name = 'tactical_dummy'
    model = tactical_model
    kwargs = {}    
    kwargs['name'] = name
    kwargs['seed'] = np.arange(2)
    kwargs['numvoters'] = 20
    kwargs['ndim'] = [1, ]
    kwargs['cnum'] = 5
    
    case_args = runtools.CaseGenerator(**kwargs)
    benchmark = runtools.CreateBenchmark(name, model, case_args)    
    return benchmark




