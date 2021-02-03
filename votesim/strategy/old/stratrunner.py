# -*- coding: utf-8 -*-
import pdb
import logging
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

logger = logging.getLogger(__name__)


def get_tactics(etype: str) -> list:
    """Retrieve tactics for a given voting method."""
    ballot_type = votemethods.get_ballot_type(etype)    
    
    if ballot_type == 'rank':
        return list(tactics_ranked.values())
    
    elif ballot_type == 'score' or ballot_type == 'rate':
        return list(tactics_scored.values())
        
    elif ballot_type == 'vote':
        return list(tactics_plurality.values())
    
    

# def get_strategies1(etype: str) -> list:
#     ballot_type = votemethods.get_ballot_type(etype)    
#     strategies = []
#     if ballot_type == 'rank':
#         for tactic_list in tactics_ranked.values():
#             strategy = {'tactics' : tactic_list,
#                         'frontrunnertype' : 'eliminate'}
#             strategies.append(strategy)
            
#     elif ballot_type == 'score' or ballot_type == 'rate':
#         for tactic_list in tactics_scored.values():
#             strategy = {'tactics' : tactic_list,
#                         'frontrunnertype' : 'eliminate'}
#             strategies.append(strategy)    
            
#     elif ballot_type == 'vote':
#         for tactic_list in tactics_plurality.values():
#             strategy = {'tactics' : tactic_list,
#                         'frontrunnertype' : 'eliminate'}
#             strategies.append(strategy)    
#     return strategies


# def get_topdog_strategy1(etype: str) -> dict:
#     """Return topdog defensive strategy"""
#     strategy = {'tactics' : ['bullet_preferred'],
#                 'frontrunnertype' : 'eliminate',
#                 'subset' : 'topdog'}
#     return strategy

# def get_strategies2(etype: str) -> list:
#     """Build election strategies given an votemethod name.
    
#     Parameters
#     ----------
#     etype : str
#         Voting method
    
#     Returns
#     -------
#     out : list[dict]
#         A list of available strategies for the method. 
        
#     """

#     keywords = votemethods.method_keywords[etype]
#     ballot_type = votemethods.get_ballot_type(etype)    
#     strategies = []
#     #######################################################################
#     if ballot_type == 'rank':
#         if 'condorcet' in keywords:
#             frontrunnertype = 'condorcet'
#         else:
#             frontrunnertype = 'tally'
#         for tactic_list in tactics_ranked.values():
#             strategy = {'tactics' : tactic_list,
#                         'frontrunnertype' : frontrunnertype,}
#             strategies.append(strategy)
            
#     #######################################################################      
#     elif ballot_type == 'score' or ballot_type == 'rate':
#         if etype == 'smith_score':
#             frontrunnertype = 'condorcet'
#         else:
#             frontrunnertype = 'tally'
        
#         for tactic_list in tactics_scored.values():
#             strategy = {'tactics' : tactic_list,
#                         'frontrunnertype' : frontrunnertype,}
#             strategies.append(strategy)   
            
#     #######################################################################        
#     elif ballot_type == 'vote':
#         for tactic_list in tactics_plurality.values():
#             strategy = {'tactics' : tactic_list,
#                         'frontrunnertype' : 'tally',}    
#             strategies.append(strategy)        
#     return strategies


# def get_topdog_strategy2(etype: str) -> dict:
#     """Return topdog defensive strategy"""
#     ballot_type = votemethods.get_ballot_type(etype)    
#     keywords = votemethods.method_keywords[etype]
#     frontrunnertype = 'tally'   
    
#     # if ballot_type == 'rank':
#     if 'condorcet' in keywords:
#         frontrunnertype = 'condorcet'
#     else:
#         frontrunnertype = 'tally'    
    
#     strategy = {'tactics' : ['bullet_preferred'],
#                 'frontrunnertype' : frontrunnertype,
#                 'subset' : 'topdog'}

#     return strategy
  
    
class StrategicRunner(object):
    def __init__(self, 
                 name: str, 
                 method : str,
                 voters: spatial.Voters,
                 candidates: spatial.Candidates,
                 ratio=1.0,
                 frontrunnernum=2,
                 user_args: dict = None,
                 record: spatial.ResultRecord = None,
                 get_all_defense=False
                 ):
        """Tactical Election model """
        
        if record is None:
            record = spatial.ResultRecord()
            
        # Construct base one-sided strategy
        strategy1 = {}
        strategy1['ratio'] = ratio
        strategy1['frontrunnernum'] = frontrunnernum
        strategy1['subset'] = 'underdog'
        
        # Construct base two-sided strategy
        strategy2 = {}
        strategy2['ratio'] = ratio
        strategy2['frontrunnernum'] = frontrunnernum
        strategy2['subset'] = 'topdog'
            
        #-------------------------------------------------------------------
        # STEP1: HONEST LECTION SIMULATION
        # Run honest election
        e = spatial.Election(voters=voters,
                             candidates=candidates,
                             strategies=None,
                             save_records=False,
                             seed=0,
                             name=name)
        e.user_data(user_args)
        
        result1 = e.run(etype=method)
        record.append(result1)
        cnum = len(candidates.data.pref)
        tactics = get_tactics(method)
        self.vse_honest = result1.stats.winner.regret_efficiency_candidate
        
        #-------------------------------------------------------------------
        # STEP 2: ONE SIDED ELECTION SIMULATIONS 
        # Perform tactics on every candidate, see which one produces benefit 
        # to the candidate's supporters.     
        logger.info('Running one-sided simulations')
        dvse_topdogs = []
        dvse_underdogs = []
        effective_list = []
        vse_list1 = []
        underdogs = []    
        for underdog in range(cnum):
            strategy1['underdog'] = underdog
    
            for tactic in tactics:
                strategy1['tactics'] = tactic
                strategies = spatial.Strategies(e.voters).add(strategy1, 0)
                e.set_models(strategies=strategies)
                result2 = e.run(etype=method, result=result1)
                winner = result2.winners[0]
                
                # Get underdog change in VSE (positive means the group benefited)
                compare = TacticCompare(e_strat=result2.stats,
                                        e_honest=result1.stats)
                dvse_u = compare.regret_efficiency_candidate['underdog-0']
                dvse_t = compare.regret_efficiency_candidate['topdog-0']
                
                # record VSE
                vse = result2.stats.winner.regret_efficiency_candidate
                vse_list1.append(vse)
                
                logger.debug('underdog #%s', underdog)
                logger.debug('strategy = %s', tactic)
                logger.debug(compare.voter_nums)
                logger.debug('underdog VSE change = %.2f', dvse_u)
                logger.debug('topdog VSE change = %.2f', dvse_t)
                try:
                    dvse_u = float(dvse_u)
                    dvse_t = float(dvse_t)
                except TypeError:
                    raise ValueError('Something went wrong with VSE calculation.')
                    
                # record tactic if it succeeds. 
                if dvse_u > 0 and winner == underdog:
                    dvse_underdogs.append(dvse_u)
                    dvse_topdogs.append(dvse_t)
                    effective_list.append(tactic)
                    underdogs.append(underdog)
                    
                    record.append(result2)
                    record.append_stat(compare)
    
        #-------------------------------------------------------------------
        # STEP 3: TWO SIDED ELECTION SIMULATIONS 
        # If a one-sided tactic succeeds, see if a defensive counter exists. 
        logger.info('Running two-sided simulations')
        vse_twosided_list = []
        for ii in range(len(effective_list)):
            
            dvse_topdog = dvse_topdogs[ii]
            strategy1['tactics'] = effective_list[ii]
            strategy1['underdog'] = underdogs[ii]
            
            vse_list2 = []
            for tactic2 in tactics:
                strategy2['tactics'] = tactic2
                strategy2['underdog'] = underdogs[ii]
                strategies = spatial.Strategies(e.voters)
                strategies.add(strategy1, 0)
                strategies.add(strategy2, 0)
                
                e.set_models(strategies=strategies)
                result3 = e.run(etype=method, result=result1)
            
                # Get change in VSE (positive means the group benefited)
                compare = TacticCompare(e_strat=result3.stats,
                                        e_honest=result1.stats)
                dvse_u = compare.regret_efficiency_candidate['underdog-0']
                dvse_t = compare.regret_efficiency_candidate['topdog-0']    
                vse = result3.stats.winner.regret_efficiency_candidate
                vse_list2.append(vse)
                
                logger.debug('tactic = %s', tactic2)
                logger.debug('underdog VSE change = %.2f', dvse_u)
                logger.debug('topdog VSE change = %.2f', dvse_t)
                
                try:
                    dvse_u = float(dvse_u)
                    dvse_t = float(dvse_t)
                except TypeError:
                    raise ValueError('Something went wrong with VSE calculation.')
                    
                if dvse_t > dvse_topdog:
                    record.append(result3)
                    record.append_stat(compare)
                    
                    # If a defensive tactic is found to completely counter 
                    # offense, break. 
                    # if (dvse_u == 0) and (get_all_defense == False):
                    #     break
                    
            vse_twosided = np.max(vse_list2)
            vse_twosided_list.append(vse_twosided)
    
        self.record = record
        self.dvse_underdogs = dvse_underdogs
        self.dvse_topdogs = dvse_topdogs
        self.vse_onesided = np.min(vse_list1)
        self.viable_underdogs = len(np.unique(underdogs))
        
        
        try:
            self.vse_twosided = np.min(vse_twosided_list)
        except ValueError:
            # If no values found that means underdog strategies were ineffective
            # and therefore two-sided runs were not performed. 
            self.vse_twosided = self.vse_honest
        self.df = record.dataframe()
        return 
    
    
    def underdog_tactic_effectiveness(self):
        """Estimate the effectiveness of underdog tactics."""
        
        tactics_u = self.df['args.strategy.0.add.strategy.tactics'].astype(str)
        tactics_t = self.df['args.strategy.1.add.strategy.tactics'].astype(str)
        
        dvse_u = self.df['']




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




