# -*- coding: utf-8 -*-
"""
Store voting method information and classification here. 

Attributes
----------
ranked_methods : dict
    Collection of ranked voting methods
scored_methods : dict
    Collection of cardinal/scored voting methods
rated_methods : dict
    Collection of cardinal methods with pre-set score ranges
vote_methods : dict
    Collection of single-mark ballots voting types, ie plurality.
    
    
    
method_keywords : dict
    Associated keywords for a voting method
    
    Possible keywords:
        
     - cardinal 
     - score
     - proportional rep
     - approval
     - ranked
     - condorcet
     
"""

from votesim.votemethods import (condorcet, irv, plurality, score, ranked)



ranked_methods = {}
ranked_methods['smith_minimax'] = condorcet.smith_minimax
ranked_methods['ranked_pairs'] = condorcet.ranked_pairs
ranked_methods['black'] = condorcet.black
ranked_methods['irv'] = irv.irv
ranked_methods['irv_stv'] = irv.irv_stv
ranked_methods['top_two'] = irv.top2runoff
ranked_methods['borda'] = ranked.borda

rated_methods = {}
rated_methods['approval100'] = score.approval100
rated_methods['approval75'] = score.approval75
rated_methods['approval50'] = score.approval50
rated_methods['approval25'] = score.approval25
rated_methods['score5'] = score.score5
rated_methods['score10'] = score.score10
rated_methods['star5'] = score.star5
rated_methods['star10'] = score.star10

scored_methods = {}
scored_methods['rrv'] = score.reweighted_range
scored_methods['seq_monroe'] = score.sequential_monroe
scored_methods['score'] = score.score
scored_methods['star'] = score.star
scored_methods['maj_judge'] = score.majority_judgment
scored_methods['smith_score'] = condorcet.smith_score

vote_methods = {}
vote_methods['plurality'] = plurality.plurality


all_methods = {}
all_methods.update(ranked_methods)
all_methods.update(scored_methods)
all_methods.update(rated_methods)
all_methods.update(vote_methods)


method_keywords = {}
method_keywords['score'] = ['cardinal', 'score',]
method_keywords['score5'] = method_keywords['score']
method_keywords['score10'] = method_keywords['score']

method_keywords['rrv'] = ['cardinal', 'proportional rep']
method_keywords['seq_monroe'] = ['cardinal', 'proportional rep']

method_keywords['star'] = method_keywords['score']
method_keywords['star5'] = method_keywords['score']
method_keywords['star10'] = method_keywords['score']
method_keywords['maj_judge'] = method_keywords['score']

method_keywords['approval'] = ['cardinal', 'approval',]
method_keywords['approval25'] = method_keywords['approval']
method_keywords['approval50'] = method_keywords['approval']
method_keywords['approval75'] = method_keywords['approval']
method_keywords['approval100'] = method_keywords['approval']



method_keywords['smith_minimax'] = ['ranked', 'condorcet']
method_keywords['black'] = ['ranked', 'condorcet']
method_keywords['ranked_pairs'] = ['ranked', 'condorcet']
method_keywords['smith_score'] = ['cardinal', 'score', 'condorcet']
method_keywords['irv'] = ['ranked', ]
method_keywords['irv_stv'] = ['ranked', 'proportional rep']
method_keywords['top_two'] = ['ranked', ]
method_keywords['borda'] = ['ranked', ]


method_keywords['plurality'] = ['vote', 'plurality']



frontrunner_calc = {}
frontrunner_calc['irv'] = ''
frontrunner_calc['black'] = 'eliminate'
frontrunner_calc['ranked_pairs'] = 'eliminate'


def get_ballot_type(etype):
    """Retrieve ballot type of the election type.
    
    Returns
    -------
    out : str
        String of either
            - 'rank'
            - 'score'
            - 'rate'
            - 'vote'
    """
    if etype in ranked_methods:
        return 'rank'
    elif etype in scored_methods:
        return 'score'
    elif etype in rated_methods:
        return 'rate'
    elif etype in vote_methods:
        return 'vote'
    
    





