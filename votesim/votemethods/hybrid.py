# -*- coding: utf-8 -*-


import numpy as np

import logging
from votesim.votemethods import tools
from votesim.votemethods import score

def score_better_balance(data: np.ndarray, numwin: int=1, max_score:int = 5):
    """Method based off Reddit post by jan_kasimi.
    
    https://www.reddit.com/r/EndFPTP/comments/lil4zz/scorebetterbalance_a_proposal_to_fix_some/

    Parameters
    ----------
    data : array shaped (a, b)
        Election voter scores, 0 to max. 
        Data of candidate ratings for each voter, with
        
           - `a` Voters represented as each rows
           - `b` Candidates represented as each column. 
              
    numwin : int
        Number of winners to consider.
        
    max_score : int, optional
        Max allowed score for the ballots. The default is 5.

    Returns
    -------
    winner : TYPE
        DESCRIPTION.
    ties : TYPE
        DESCRIPTION.
    output : TYPE
        DESCRIPTION.

    """
    tally = data.sum(axis=0)
    # STEP 1: Get scores
    score_winners, score_ties = tools.winner_check(tally, numwin=1)
    score_winner = score_winners[0]
    
    score_margin = (tally[score_winner] - tally) / max_score
    
    # STEP 2: Get candidates that can beat score winner head-to-head
    winner_data = data[:, score_winner : score_winner + 1]
    head2head_count_wins = data < winner_data
    head2head_count_losses = data > winner_data
    
    count_wins = head2head_count_wins.sum(axis=0)
    count_losses = head2head_count_losses.sum(axis=0)
    count_margin = count_wins - count_losses
    
    # STEP 3: Combine head-to-head margin with score margin.     
    combined_margin = count_margin - score_margin
    
    winner, ties = tools.winner_check(combined_margin, numwin=numwin)
    
    output = {}
    output['scores'] = tally
    output['win_loss_margin'] = count_margin
    output['combined_margin'] = combined_margin
    
    return winner, ties, output
    