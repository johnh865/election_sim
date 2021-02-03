# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 11:40:00 2020

@author: John
"""

import numpy as np

import logging
from votesim.votemethods import tools
logger = logging.getLogger(__name__)


def v321(data, numwin: int=1, rng=None, seed=0):
    data = np.array(data)
    
    # Convert to 3 ratings good/ok/bad
    dmax = data.max()
    data = data / dmax * 3
    data = np.round(data)
    cnum = data.shape[1]
    
    # Get score data
    sums = np.sum(data, axis=0)
    
    # ROUND 1
    # Get 3 semi-finalists with most "good" ratings
    if cnum > 3:
        tally3 = np.sum(data == 3, axis=0)
        semifinalists, semi_ties = tools.winner_check(tally3, numwin=3)
        
        # tie break        
        semifinalists, _ = v321_tie_resolver(semifinalists,
                                          semi_ties,
                                          sums,
                                          numwin=3,
                                          rng=rng,
                                          seed=seed,
                                          use_random=True)
    else:
        semifinalists = np.array([0, 1, 2])
        
    # ROUND 2
    # Get 2 finalists with the fewest "bad" ratings
    if cnum > 2:
        bad_count = np.sum(data == 0, axis=0)
        bad_count = bad_count[semifinalists] 
        finalists , fties = tools.winner_check_named(-bad_count, semifinalists)
        finalists = v321_tie_resolver(finalists, 
                                      fties,
                                      sums,
                                      numwin=2,
                                      rng=rng,
                                      seed=seed+1,
                                      use_random=True)
    else:
        finalists = np.array([0, 1])
        semifinalists = np.array([0, 1])
    
    # ROUND 3
    # Get winner who is rated above the other on more ballots
    votes1 = data[:, finalists[0]] > data[:, finalists[1]] 
    votes2 = data[:, finalists[0]] < data[:, finalists[1]] 
    final_votes = [votes1, votes2]
    winners, ties = tools.winner_check_named(final_votes, finalists)
    
    output = {}
    output['sums'] = sums
    output['semifinalists'] = semifinalists
    output['finalists'] = finalists
    
    return winners, ties, output

    
    



def v321_tie_resolver(winners, ties, sums, numwin, rng=None, seed=0,
                      use_random=True):
    """Resolve v321 ties.  
    

    Parameters
    ----------
    winners : (a,) array
        Winners found for the current round.
    ties : (b,) array
        Tie candidates found for the current round.
    sums : (cnum,) array
        Net scores for all candidates.
    numwin : int
        Number of winners for the current round.
    rng : numpy.random.Generator , optional
        Random number generator for tie breaking. The default is None.
    seed : int, optional
        Input seed into random number generator if `rng` not provided.
        The default is 0.
    use_random : bool, optional
        Use random tie resolution. The default is True. If false, ties 
        are returned after the score tie break. 

    Returns
    -------
    winners : (c,) array
        Winners after tie breaks.
    ties : (d,) array
        Remaining ties after attemped tie breaking.
    """
    
    # Resolve ties using score voting
    if len(ties) > 1:
        tie_winners, ties2 = tools.winner_check_named(sums[ties], ties)
    
    # If tie not resolved, resort to random
    if use_random:
        semi_num = len(winners) + len(tie_winners)
        if len(semi_num) < numwin:
            if rng is None:
                rng = np.random.default_rng(seed)
            
            rng.shuffle(ties2)
            needed = numwin - semi_num
            ties2 = ties2[0 : needed]
            tie_winners = np.append(tie_winners, ties2)    
        
        winners = np.append(winners, tie_winners)
        assert len(winners) == numwin
        return winners, np.array([], dtype=int)
    else:
        return tie_winners, ties2


    