# -*- coding: utf-8 -*-

"""
Condorcet voting methods.
"""
import logging
import numpy as np
from votesim.votemethods.tools import (
    winner_check,
    multi_win_eliminate_decorator,
)
import votesim.votemethods.condcalcs as condcalcs
from votesim.votemethods.ranked import borda
from votesim.votemethods.condcalcs import (smith_set,
                                           has_cycle,
                                           pairwise_rank_matrix,
                                           condorcet_winners_check,
                                           VoteMatrix
                                           )

logger = logging.getLogger(__name__)

__all__ = ['smith_minimax', 
           'ranked_pairs', 
           'smith_score']


def smith_minimax(ranks=None, numwin=1, matrix=None):
    """Condorcet Smith Minimax voting algorithm for ranked ballots.

    Parameters
    ----------
    ranks : (a, b) array 
        Election voter rankings, from [1 to b].
        Data composed of candidate rankings, with

           - Voters as the rows
           - Candidates as the columns.

        Use 0 to specify unranked (and therefore not to be counted) ballots.

        - a : number of voters dimension. Voters assign ranks for each candidate.
        - b : number of candidates. A score is assigned for each candidate
              from 0 to b-1.
    matrix : (b, b) array 
        Win-loss matrix
        
    Returns
    -------
    winners : array of shape (numwin,)
        Winning candidates index.
    ties : array of shape (tienum,)
        Tied candidates index for the last round, numbering `tienum`.
    """
    m = None
    if ranks is not None:
        m = pairwise_rank_matrix(ranks)
        win_losses = m - m.T
    elif matrix is not None:
        win_losses = np.array(matrix)

    cnum = len(win_losses)

    s = smith_set(wl=win_losses)
    s = list(s)

    ifail = np.ones(cnum, dtype=bool)
    ifail[s] = False

    min_losses = np.min(win_losses, axis=1)
    min_min = np.min(min_losses)
    min_losses[ifail] = min_min - 10

    #candidates = np.arange(cnum)
    #imax = np.argmax(min_losses[s])
    #winner = candidates[s][imax]

    winners, ties = winner_check(min_losses, numwin=1)
    
    output = {}
    if m is not None:
        output['margin_matrix'] = win_losses
        output['vote_matrix'] = m
    output['tally'] = min_losses
    output['smith_set'] = s
    return winners, ties, output


@multi_win_eliminate_decorator
def ranked_pairs(ranks, numwin=1,):
    """Ranked-pairs or Tideman election system.

    Parameters
    ----------
    ranks : (a, b) array 
        Election voter rankings, from [1 to b].
        Data composed of candidate rankings, with

           - Voters as the rows
           - Candidates as the columns.

        Use 0 to specify unranked (and therefore not to be counted) ballots.

        - a : number of voters dimension. Voters assign ranks for each candidate.
        - b : number of candidates. A score is assigned for each candidate
              from 0 to b-1.

    Returns
    -------
    winners : (numwin,) array 
        Winning candidates index.
    ties : (tienum,) array 
        Tied candidates index for the last round, numbering `tienum`.
    output : dict
        Dictionary with additional election data

        pairs : (a, 3) array 
            Win-Loss candidate pairs
            - column 0 = winning candidate
            - column 1 = losing candidate
            - column 2 = margin of victory

        locked_pairs : (b, 3) array 
            Win-Loss candidate pairs, with low margin-of-victory winners
            who create a cycle eliminated.
            - column 0 = winning candidate
            - column 1 = losing candidate
            - column 2 = margin of victory
    """
    m = pairwise_rank_matrix(ranks)
    win_losses = m - m.T
    cnum = len(m)

    # construct win-loss candidate pairs
    pairs = []
    for i in range(cnum):
        for j in range(cnum):
            winlosses = win_losses[i, j]
            if winlosses > 0:
                d = [i, j, winlosses]
                pairs.append(d)

    # Sort the pairs with highest margin of victory first.
    pairs = np.array(pairs)
    locked_pairs = []

    logger.debug('win matrix=\n%s', m)
    logger.debug('pairs=\n%s', pairs)

    # Tied if no win pairs found
    if len(pairs) == 0:
        winners = np.array([], dtype=int)
        ties = np.arange(cnum, dtype=int)
        scores = np.zeros(cnum, dtype=int)
    else:
        i = np.argsort(pairs[:, 2])[::-1]
        pairs = pairs[i]

        # Eliminate low margin pairs that create a cycle
        locked_pairs.append(pairs[0])
        for pair in pairs[1:]:
            ipairs = np.row_stack((locked_pairs, pair))
            is_cycle = has_cycle(ipairs,)
            if not is_cycle:
                locked_pairs = ipairs
                logger.debug('locking in pair %s', pair)
            else:
                logger.debug('cycle found for pair %s', pair)

        # Find the condorcet winner in the locked candidate pairs
        winners, ties, scores = condorcet_winners_check(
            pairs=locked_pairs,
            numwin=1,
        )

    # Handle some additional output
    output = {}
    output['pairs'] = pairs
    output['locked_pairs'] = locked_pairs
    output['margin_matrix'] = win_losses
    output['vote_matrix'] = m
    output['win_rankings'] = scores

    return winners, ties, output


def ranked_pairs_test2(ranks, numwin=1):
    output = {}
    vm = VoteMatrix(ranks=ranks)
    pairs = vm.pairs
    winners, ties, scores = condorcet_winners_check(pairs=pairs)



    if len(winners) == numwin:
        output['pairs'] = pairs
        output['locked_pairs'] = pairs
        output['margin_matrix'] = vm.margin_matrix
        output['win_rankings'] = scores
        return winners, ties, output

    pairs = np.array(pairs)
    i = np.argsort(pairs[:, 2])[::-1]
    pairs = pairs[i]

    # Eliminate low margin pairs that create a cycle
    locked_pairs = []
    locked_pairs.append(pairs[0])
    for pair in pairs[1:]:

        ipairs = np.row_stack((locked_pairs, pair))
        is_cycle = has_cycle(ipairs,)
        if not is_cycle:
            locked_pairs.append(pair)

    locked_pairs = np.array(locked_pairs)
    winners, ties, scores = condorcet_winners_check(pairs=pairs)
    output['pairs'] = pairs
    output['locked_pairs'] = locked_pairs
    output['margin_matrix'] = vm.margin_matrix
    output['win_rankings'] = scores
    return winners, ties, output


#
#def detect_cycles(pairs, maxiter=1000):
#    """
#    Detect if condorcet cycle exists for pairs of candidate win-loss margins
#
#    Parameters
#    ----------
#    pairs : array shaped (a, 3)
#        Win-Loss candidate pairs
#        - column 0 = winning candidate
#        - column 1 = losing candidate
#        - column 2 = margin of victory
#
#    Returns
#    -------
#    out : bool
#        True if condorcet cycle detected. False otherwise.
#    """
#
#    c = _CycleDetector(pairs, maxiter=maxiter)
#    return c.any_circuits()
#
#
#


@multi_win_eliminate_decorator
def smith_score(data, numwin=1,):
    """Smith then score voting variant.
    
    Parameters
    ----------
    data : array shaped (a, b)
        Election voter scores, 0 to max. 
        Data of candidate ratings for each voter, with
        
           - `a` Voters represented as each rows
           - `b` Candidates represented as each column. 
              
    numwin : int
        Number of winners to consider
    
    Returns
    -------
    winners : (numwin,) array
        Winning candidates index.
    ties : (tienum,) array 
        Tied candidates index for the last round, numbering 'tienum'.
    output : dict    
        sums : (b,) array
            Score sums for each candidate
        vote_matrix : (b,b) array
            Head-to-head count where row-wise candidate score beats
            colum-wise candidate.
        smith_set : (a,) array
            Candidate indices who are within the Smith Set. 
    """
    data = np.atleast_2d(data)
    sums = data.sum(axis=0)
    cnum = data.shape[1]


    in_smith = np.zeros(cnum, dtype=bool)

    m = condcalcs.pairwise_scored_matrix(data)
    smith = condcalcs.smith_set(vm=m)
    smith = list(smith)

    in_smith[smith] = True
    sums[~in_smith] = 0

    winners, ties = winner_check(sums, numwin=1)

    output = {}
    output['sums'] = sums
    output['vote_matrix'] = m
    output['smith_set'] = smith
    output['margin_matrix'] = m - m.T
    return winners, ties, output



@multi_win_eliminate_decorator
def black(ranks):
    """Condorcet-black."""
    m = pairwise_rank_matrix(ranks)
    win_losses = m - m.T
    winners, ties, scores = condorcet_winners_check(matrix=win_losses)
    output = {}
    output['margin_matrix'] = win_losses    
    output['tally'] = None
    if len(winners) > 0:
        return winners, ties, output    
    else:
        winners, ties, b_output = borda(ranks, numwin=1)
        output.update(b_output)
        return winners, ties, output
        
    
    
    
    
    
    
    






