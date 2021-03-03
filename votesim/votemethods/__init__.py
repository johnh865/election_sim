# -*- coding: utf-8 -*-
"""
Repository of voting methods. All voting methods here have the same
function interface and are interchangeable. 



Attributes
-----------
ranked_methods : dict
    Collection of ranked voting methods with their name as the key and 
    the voting system's function as the value.
scored_methods : dict
    Collection of ranked voting methods with their name as the key and 
    the voting system's function as the value.
vote_methods : dict
    Collection of single-mark voting methods with their name as the key and 
    the voting system's function as the value.    
all_methods : dict
    Collection of all available voting methods with their name as the key and 
    the voting system's function as the value.    
            
"""

# from votesim.votemethods.methodinfo import (
#     # ranked_methods, 
#     # rated_methods,
#     # scored_methods,
#     # vote_methods, 
#     # all_methods,    
#     # method_keywords,
#     # get_ballot_type,
# )
from votesim.votemethods import (irv,
                                 plurality,
                                 score,
                                 ranked,
                                 tools,
                                 condorcet,
                                 condcalcs,
                                 )


TYPE_RANK = 'rank'
TYPE_SCORE = 'score'
TYPE_RATE = 'rate'
TYPE_VOTE = 'vote'

SMITH_MINIMAX = 'smith_minimax'
RANKED_PAIRS = 'ranked_pairs'
BLACK = 'black'
COPELAND = 'copeland'
IRV = 'irv'
IRV_STV = 'irv_stv'
STV_GREGORY = 'stv_gregory'
TOP_TWO = 'top_two'
BORDA = 'borda'

ranked_methods = {}
ranked_methods[SMITH_MINIMAX] = condorcet.smith_minimax
ranked_methods[RANKED_PAIRS] = condorcet.ranked_pairs
ranked_methods[BLACK] = condorcet.black
ranked_methods[IRV] = irv.irv
ranked_methods[IRV_STV] = irv.irv_stv
ranked_methods[STV_GREGORY] = irv.stv_gregory
ranked_methods[TOP_TWO] = irv.top2runoff
ranked_methods[BORDA] = ranked.borda
ranked_methods[COPELAND] = condorcet.copeland

SCORE = 'score'
STAR = 'star'
REWEIGHTED_RANGE = 'rrv'
SEQUENTIAL_MONROE = 'seq_monroe'
MAJORITY_JUDGMENT = 'maj_judge'
SMITH_SCORE = 'smith_score'
PLURALITY = 'plurality'

scored_methods = {}
scored_methods[REWEIGHTED_RANGE] = score.reweighted_range
scored_methods[SEQUENTIAL_MONROE] = score.sequential_monroe
scored_methods[SCORE] = score.score
scored_methods[STAR] = score.star
scored_methods[MAJORITY_JUDGMENT] = score.majority_judgment
scored_methods[SMITH_SCORE] = condorcet.smith_score

APPROVAL100 = 'approval100'
APPROVAL75 = 'approval75'
APPROVAL50 = 'approval50'
APPROVAL25 = 'approval25'
SCORE5 = 'score5'
SCORE10 = 'score10'
STAR5 = 'star5'
STAR10 = 'star10'

rated_methods = {}
rated_methods[APPROVAL100] = score.approval100
rated_methods[APPROVAL75] = score.approval75
rated_methods[APPROVAL50] = score.approval50
rated_methods[APPROVAL25] = score.approval25
rated_methods[SCORE5] = score.score5
rated_methods[SCORE10] = score.score10
rated_methods[STAR5] = score.star5
rated_methods[STAR10] = score.star10


vote_methods = {}
vote_methods[PLURALITY] = plurality.plurality


all_methods = {}
all_methods.update(ranked_methods)
all_methods.update(scored_methods)
all_methods.update(rated_methods)
all_methods.update(vote_methods)

# eRunner is reliant on some of dict definitions. Import after.
from votesim.votemethods.voterunner import eRunner



def get_ballot_type(etype: str):
    """Retrieve ballot type of the election type.
    
    Parameters
    ----------
    etype : str
        Election method name, see `all_methods.keys()` for all options.  
    
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
        return TYPE_RANK
    
    elif etype in scored_methods:
        return TYPE_SCORE
    
    elif etype in rated_methods:
        return TYPE_RATE
    
    elif etype in vote_methods:
        return TYPE_VOTE
    
    














