# -*- coding: utf-8 -*-
"""
Module for instant runoff voting. 

Created on Sun Oct 13 22:11:08 2019

@author: John
"""

import numpy as np
import logging
from votesim.votemethods.tools import (rcv_reorder,
                                       droop_quota, 
                                       winner_check)

__all__ = [
    'irv',
    'irv_eliminate',
    'irv_stv',
    'top2runoff']
logger = logging.getLogger(__name__)
### RANKED CHOICE / SINGLE TRANSFERABLE VOTE 


def  hare_quota(votes, seats):
    return votes / seats





def top2runoff(data, numwin=1):
    """
    Emulate top two runoff using ranked data

    
    Parameters
    ----------
    data : array shaped (a, b)
        Election voter rankings, from [1 to b].
        Data composed of candidate rankings, with
        
           - Voters as the rows, with `a` total voters
           - Candidates as the columns, with `b` total candidates.
           
        Use 0 to specify unranked (and therefore not to be counted) ballots.  
        
    numwin : int
        Number of winners
        
    Returns
    --------
    winners : array of shape (numwin,)
        Winning candidates index.
    ties : array of shape (tienum,)
        Tied candidates index for the last round, numbering 'tienum'.
    output : dict
        talley : array shape (b,)
            Number of transferred votes obtained by candidates before elimination.
       
    """
    
    ### round #1
    if numwin > 1:
        raise ValueError('Only numwinner=1 supported')
        
    data = np.array(data)
    candidate_num = data.shape[1]
    
    vote_index = data == 1
    vote_count = np.sum(vote_index, axis=0)
    winners, ties = winner_check(vote_count, numwin=2)
    winners = np.append(winners, ties)
    
    
    loser_bool = np.ones(candidate_num, dtype=bool)
    loser_bool[winners] = False
    
    # zero out all losers
    data[:, loser_bool] = 0
    data = rcv_reorder(data)
            
    ### round #2
    vote_index = data == 1
    vote_count2 = np.sum(vote_index, axis=0)
    winners2, ties2 = winner_check(vote_count2, numwin=1)
    
    output = {}
    output['tally'] = np.maximum(vote_count, vote_count2)
    output['runoff_candidates'] = winners
    output['runoff_tally'] = vote_count2[winners]
    output['first_tally'] = vote_count
    
    return winners2, ties2, output


   


def irv(data, numwin=1, seed=None):
    """
    Calculate winners of an election using Instant Runoff Voting.
    Break 1st place ties using 2nd, 3rd, etc ranks.
    If ties still found at last place rank, return ties. 
    
    Parameters
    ----------
    data : array shaped (a, b)
        Election voter rankings, from [1 to b].
        Data composed of candidate rankings, with
        
           - Voters as the rows, with `a` total voters
           - Candidates as the columns, with `b` total candidates.
           
        Use 0 to specify unranked (and therefore not to be counted) ballots.  
        
    numwin : int
        Number of winners
        
    Returns
    --------
    winners : array of shape (numwin,)
        Winning candidates index.
    ties : array of shape (tienum,)
        Tied candidates index for the last round, numbering 'tienum'.
    output : dict
        
        round_history : array of shape (numwin, b)
            Vote counts for each candidate, for each round.        
        data : array of shape (a, b)
            Re-ordered ranking data after losers have been eliminated,
            retaining winner ranksings.
        talley : array shape (b,)
            Number of transferred votes obtained by candidates before elimination.
    """
    rstate = np.random.RandomState(seed)
#    if rstate is None:
#        rstate = np.random.RandomState()    
    
    data = np.array(data, copy=True)    
    
    # Only consider ballots that are not blank
    ii_filled = np.sum(data, axis=1) > 0
    data = data[ii_filled]
    
    voter_num, candidate_num = data.shape
    numrounds = max(1, candidate_num - numwin)
        
    logger.debug('# of rounds = %s', numrounds)
    logger.debug('# of winners = %s', numwin)
    
    # initialize history datas
    round_history = []
    loser_log = []
    
    start_losers = np.sum(data, axis=0) == 0
    winners_bool = np.ones(candidate_num, dtype=bool)
    winners_bool[start_losers] = False
    
    for i in range(numrounds):
        
        logger.debug('irv round # %d', i)
        logger.debug('data\n%s', data)
        
        # Fine eliminated candidate
        data, loser, ties, history = irv_eliminate(data)
        try:
            round_history.append(history[0])
        except KeyError: 
            pass
        
        tienum = len(ties)
        
        logger.debug('Losers=%s', loser)
        loser_log.append(loser)
        
        if tienum > 1:
            # Break low-level ties via random number. 
            if i < numrounds - 1:
                kk = rstate.randint(0, tienum)
                loser = ties[kk]
                data[:, loser] = 0      
                logger.debug('Random tie break, candidate %s', loser)
        
            # If last round, ties shall be passed out of the function.
            # Set loser to ties to make sure they're not passed out as winners
            elif i == numrounds - 1:
                # loser = ties
                winners_bool[ties] = False
        
        if loser == -1:
            pass
        else:
            winners_bool[loser] = False   
            
        survivors = np.where(winners_bool)[0]
        logger.debug('Survivors=%s', survivors)           
        if len(survivors) <= numwin:
            break

    winners = survivors
    round_history = np.array(round_history)
    
    logger.debug('winners=%s', winners)
    logger.debug('ties=%s', ties)    
    
    
    tally = np.nanmax(round_history, axis=0)
    tally[np.isnan(tally)] = 0
    
    output = {}
    output['tally'] = tally
    output['round_history'] = round_history
    output['loser_history'] = np.array(loser_log)
    # output['data'] = data
    
    return winners, ties.astype(int), output
            
      
        
        

def irv_eliminate(data):
    """Eliminate a candidate using ranked choice, instant runoff voting.
    
    Parameters
    ----------
    data : array shaped (a, b)
        Election voter rankings, from [1 to b].
        Data composed of candidate rankings, with
        
           - Voters as the rows, with `a` total voters
           - Candidates as the columns, with `b` total candidates.
           
        Use 0 to specify unranked (and therefore not to be counted) ballots.  
        

    Returns
    -------
    data : array shaped (a, b)
        Election voter rankings, but with losing/eliminated candidate's data zeroed. 
    loser : int 
        Column integer index of data of losing candidate. 
        
        - lower will be equal to -1 if no candidates can be eliminated.
        - loser will be 0 or larger, if a candidate can be eliminated.
    ties : array shaped (c,)
        Index locations of tied losing candidates. Empty array if no ties.
        
    history : array shaped (n, b)
        History of each elimination round. Multiple rounds will occur if 
        ties are found during elimination. `n` is number of rounds. 
    """   
    data = np.array(data, copy=True)    
    candidate_num = data.shape[1]
    start_losers = np.sum(data, axis=0) == 0
    losernum = np.sum(start_losers)
    round_history = []
    
    logger.debug('irv elimination start.')
    logger.debug('# losers = %s', losernum)
                
    active_bool = np.ones(candidate_num, dtype=bool)    
    active_bool[start_losers] = False
    tie_bool = np.zeros(candidate_num, dtype=bool)    
    data2 = data.copy()
    
    # Handle all zeros array 
    if np.all(start_losers):
        logger.debug('All zero data input into irv eliminate')
        ties = np.array([], dtype=int)
        loser = -1
        vote_index = (data2 == 1)
        vote_count = np.sum(vote_index, axis=0, dtype='float64')
        round_history.append(vote_count)
        return data, loser, ties, np.array(round_history)
    
    elif (candidate_num - losernum) == 1:
        logger.debug('All but one candidate already eliminated')
        ties = np.array([], dtype=int)
        loser = -1
        vote_index = (data2 == 1)
        vote_count = np.sum(vote_index, axis=0, dtype='float64')
        round_history.append(vote_count)        
        return data, loser, ties, np.array(round_history)

    
    for j in range(1, candidate_num + 1):
        
        vote_index = (data2 == j)
        vote_count = np.sum(vote_index, axis=0, dtype='float64')
        round_history.append(vote_count)
        
        vote_count[~active_bool] = np.nan
        
        # Negative votes to get loser rather than winner.
        losers, ties = winner_check(-vote_count, numwin=1)
        logger.debug('Eliminating from rank #%d', j)
        logger.debug('\n,%s', data2)
        logger.debug('count=%s', vote_count)
        logger.debug('losers = %s', losers)
        logger.debug('ties = %s', ties)
        
        if len(ties) == 0:
            loser = losers[0]
            break
        else:
            loser = -1
            tie_bool[ties] = True
            active_bool = tie_bool
            
    if len(ties) == 0:        
        # Zero out loser rank data
        data[:, loser] = 0
        data = rcv_reorder(data)     
    return data, loser, ties, np.array(round_history)
        

    
def irv_stv(data, numwin=1, reallocation='hare',
            weights=1, seed=None, maxiter=500):    
    """
    Calculate winners of an election using Single Transferable Vote
    
    Parameters
    ----------
    data : array shaped (a, b)
        Election voter rankings, from [1 to b].
        Data composed of candidate rankings, with
        
           - Voters as the rows
           - Candidates as the columns. 
           
        Use 0 to specify unranked (and therefore not to be counted) ballots.  
        
        - a : number of voters dimension. Voters assign ranks for each candidate. 
        - b : number of candidates. A score is assigned for each candidate 
              from 0 to b-1.   
              
    numwin : int
        Number of winners
    reallocation : str
        Vote reallocation algorithm for vote transfer. 
        
        - 'hare' -- randomized transfer of surplus votes.
        - 'gregory' -- Weighted transfer of surplus votes.
        
    weights : array shaped (a,), int, or float
        Initial ballot weights, only works on gregory for now. 
        
    seed : int or Nont
        Set pseudo-random number generator for Hare.
    
    Returns
    -------
    winners : int array shaped (numwin,)
        Candidate index locations of winners. 
    ties : int array shaped (c,)    
        Candidate index location of tied candidates. 
    history : int array shaped (d, b)
        Vote counting record for each round. 
    """
    rstate = np.random.RandomState(seed)
    
    # retrieve number of filled ballots. Omit blank ballots.
    data = _prep_data(data)
    num_ranked, num_candidates = data.shape
    
    quota = droop_quota(num_ranked, numwin)
    logger.info('stv droop quota = %s', quota)
    
    if reallocation=='hare':
        allocate = hare_reallocation
        
    elif reallocation == 'gregory':
        allocate = gregory_reallocation
        
    else:
        raise ValueError(reallocation + ' not valid reallocation method')    
        
    round_history = []
    winners = []
    exhausted = []
    winner_count = 0
    survivor_count = num_candidates
    survivor_bool = np.ones(num_candidates, dtype=bool)
    
    for ii in range(maxiter):    
        # Get this round's winners from votes exceeding quota.
        vote_index = (data == 1)
        
        # Get tally totals for each candidate * weights
        tally = np.sum(vote_index * weights, axis=0)     
        round_history.append(tally)
        
        winners_ii = np.where(tally >= quota)[0]
        winners = np.append(winners, winners_ii)
        winner_count = len(winners)
        
        # Get voters who won
        
        
        logger.info('\n\nstv round #%d', ii)
        logger.info('stv votes = %s', tally)
        logger.info('stv winners = %s', winners)
        
        # Break if we've gotten all winners        
        if winner_count >= numwin:
            ties = np.array([])
            break
        
        # If winner found, reallocate surplus votes
        if len(winners_ii) > 0:
            data, weights = allocate(
                data = data, 
                tally = tally,
                winners = winners_ii, 
                quota = quota,
                weights=weights,
                rstate=rstate
            )
        # Perform instant runoff counting & elimination
        # irv eliminates candidates by zeroing out their rank data.
        else:
            data, exhaustedi, ties, hist = irv_eliminate(data)
            
            # Randomly eliminate loser if tie found. 
            tienum = len(ties)
            tied = tienum > 1
            
            if tied:
                logger.warning('Ties found for IRV elimination for %s', ties)
                
                # Check if there are too many ties to complete STV
                if winner_count + tienum > numwin + survivor_count - tienum:
                    logger.warning('Ties %s too close to winner. Outputting ties', ties)
                    break
                else: 
                    jj = rstate.randint(0, tienum)
                    exhaustedi = ties[jj]
                    data[:, exhaustedi] = 0
                    data = rcv_reorder(data)                
                    logger.warning('Randomly eliminated ties %s', exhaustedi)

            exhausted.append(exhaustedi)
            exhausted_count = len(exhausted)            
            survivor_count = num_candidates - exhausted_count
            
        ######################################################################
        # check for special condition if survivors equal number of winners left. 
        if survivor_count + winner_count <= numwin:
            
            survivor_bool[np.array(exhausted)] = False
            survivors = np.where(survivor_bool)[0]
            winners = np.append(winners, survivors)

            logger.warning(
                'Too few survivors = %s, %s winners',
                survivor_count,
                winner_count,
            )
            break        
        
        logger.debug('stv survived # = %s', survivor_count)
        logger.debug('stv exhausted = %s', exhausted)         
        
    winners = winners.astype(int)
    round_history = np.row_stack(round_history)
    
    output = {}
    output['round_history'] = round_history
    output['quota'] = quota
    return winners, ties, output


def _prep_data(data):
    data = np.atleast_2d(data).copy()
    data = rcv_reorder(data)     
    locs_filled = np.max(data > 0, axis=1)
    data = data[locs_filled]
    return data
        
        
def stv_gregory(data, numwin=1, weights=1, maxiter=500):
    return irv_stv(data, numwin=numwin,
                   reallocation='gregory',
                   weights=weights, maxiter=maxiter)

    
    
def hare_reallocation(data, tally, winners, quota, weights, rstate=None):
    """
    Re-allocate ranked data by random.
    
    Parameters
    ----------
    data : array shaped (a, b)
        Election voter rankings, from [1 to b].
        Data composed of candidate rankings, with
        
           - Voters as the rows, with `a` total voters
           - Candidates as the columns, with `b` total candidates.
           
        Use 0 to specify unranked (and therefore not to be counted) ballots.  
        
    winners : array shaped (c,)
        The round winners's data column indices
    
    quota : int
        STV winning quota
    rstate : numpy random.RandomState or None (default)
        Random number generating object.
        
    Returns
    -------
    data : array shaped (a, b)
        Election voter rankings, with winning candidates surplus votes transferred
        to runner ups. 
    """
    
    if rstate is None:
        rstate = np.random.RandomState()
                
    for ii, winner in enumerate(winners):
        
        win_voter_locs = data[:, winner] == 1
        win_voter_index = np.flatnonzero(win_voter_locs)
        
        vote_num = tally[ii]
        
        # Remove ballots the size of the quota
        remove_index = rstate.choice(
            win_voter_index,
            size = min(vote_num, quota),
            replace = False,
            )
        
        data = np.delete(data, remove_index, axis=0)
        
        # Zero out the winner
        data[:, winner] = 0
        
    data = rcv_reorder(data)
    return data, 1.0

    
    
def gregory_reallocation(
        data: np.ndarray,
        tally: np.ndarray, 
        winners: np.ndarray,
        quota: int,
        weights: np.ndarray,
        **kwargs):
    
    voter_num = len(data)
    
    for ii, winner in enumerate(winners):
        win_voter_locs = np.flatnonzero(data[:, winner] == 1)
        win_num = tally[ii]
        surplus_factor = (win_num - quota) / win_num
        
        factors = np.ones((voter_num, 1))
        factors[win_voter_locs] = surplus_factor
        weights = factors * weights 
        pass
    
    return data, weights
    

def score2rank(data):
    i = np.argsort(np.argsort(-data, axis=1),axis=1) + 1
    return i
        


def ____BROKEN_IRV_eliminate(data):
    """Eliminate a candidate using ranked choice, instant runoff voting.
    
    Parameters
    ----------
    data : array shaped (a, b)
        Election voter rankings, from [1 to b].
        Data composed of candidate rankings, with
        
           - Voters as the rows, with `a` total voters
           - Candidates as the columns, with `b` total candidates.
           
        Use 0 to specify unranked (and therefore not to be counted) ballots.  
        

    Returns
    -------
    data : array shaped (a, b)
        Election voter rankings, but with losing/eliminated candidate's data zeroed. 
    loser : int
        Column integer index of data of losing candidate. 
        
        - lower will be equal to -1 if no candidates can be eliminated.
        - loser will be 0 or larger, if a candidate can be eliminated.
    ties : array shaped (c,)
        Index locations of tied losing candidates. Empty array if no ties. 
    """
    
    data = np.copy(data)
    num_candidates = data.shape[1]    
    num_voters = data.shape[0]
        
    safe_candidates = np.zeros(num_candidates, dtype=bool)
    
    # iterate through rankings until a rank is found with no ties. 
    for rank in range(1, num_candidates + 1):
        logger.info('Eliminating for rank %s' % rank)
        index = (data == rank)
        vote_totals = index.sum(axis=0)
        if rank == 1:
            # get previously eliminated candidates
            eliminated = np.sum(data, axis=0) == 0
            logger.info('Previously eliminated candidates=%s', np.where(eliminated)[0])
            
            # Add eliminated to the safe list
            safe_candidates = safe_candidates | eliminated

        logger.info('Vote totals')
        logger.info(vote_totals)
              
        # Ignore safe candidates
        vote_totals[safe_candidates] = num_voters
        proposed_loser = np.argmin(vote_totals)
        
        # check for loser ties
        tie_bools = vote_totals[proposed_loser] == vote_totals
        tie_num = np.sum(tie_bools)
        
        # update safe candidates to include anyone not tied with loser. 
        safe_candidates = safe_candidates | (~tie_bools)        
        logger.info('Safe & eliminated candidates = %s', np.where(safe_candidates)[0])
        logger.info('Unsafe candidates = %s', np.where(~safe_candidates)[0])
        
        # If no ties, eliminate
        if tie_num == 1:
            loser = proposed_loser
            logger.info('Eliminating candidate %s using #%s ranking' % (loser, rank) )
            
            # Completely zero out the loser on all ballots. 
            data[:, loser] = 0
            
            # Ensure ranking order is good
            data = rcv_reorder(data)
            return data, loser, np.array([])
        
        # Continue iteration if ties found.
        elif tie_num > 1:
            logger.info('Elimination tie found for candidates, at ranking #%s' % rank)
            logger.info(np.where(tie_bools)[0])            
            
    logger.info('Tie found with no tie breaker.')
    tie_index = np.where(tie_bools)[0]
    return data, -1, tie_index
    
        
    

def ___BROKEN_IRV_STV(data, numwin=1, reallocation='hare', maxiter=50, rstate=None):
    """
    Calculate winners of an election using Single Transferable Vote
    
    Parameters
    ----------
    data : array shaped (a, b)
        Election voter rankings, from [1 to b].
        Data composed of candidate rankings, with
        
           - Voters as the rows
           - Candidates as the columns. 
           
        Use 0 to specify unranked (and therefore not to be counted) ballots.  
        
        - a : number of voters dimension. Voters assign ranks for each candidate. 
        - b : number of candidates. A score is assigned for each candidate 
              from 0 to b-1.   
              
    numwin : int
        Number of winners
    reallocation : str
        Vote reallocation algorithm for vote transfer. 
        
        - 'hare' -- randomized transfer of surplus votes.
        
    rstate : None or `numpy.random.RandomState`
        Set to None to be truly random.
        Set RandomState to use deterministic pseudo-random number generator
    
    Returns
    -------
    winners : int array shaped (numwin,)
        Candidate index locations of winners. 
    ties : int array shaped (c,)    
        Candidate index location of tied candidates. 
    history : int array shaped (d, b)
        Vote counting record for each round. 
    """
    if rstate is None:
        rstate = np.random.RandomState
    
    data = np.atleast_2d(data).copy()
    data = rcv_reorder(data)
    dmax = data.max()
    
    
    original = np.copy(data)
    
    num_candidates = data.shape[1]
    num_voters = data.shape[0]
    quota = droop_quota(num_voters, numwin)
    
    logger.info('quota = %d' % quota)
    logger.info('quota percent = %7.3f' % (quota / num_voters))
    

    winner_list = []
    round_results = []
    ties = np.array([])
    i = 0
    while len(winner_list) < numwin:
        i += 1
        if i >= maxiter:
            logger.info('MAX ITERATION OF %s REACHED. TERMINATING' % maxiter)
            break
        
        # get the votes for the i^th round, shaped (a, b)
        ith_round_votes = (data == 1)        
        
        # total votes for each candidate; array shaped (b); number of candidates
        ith_vote_totals = ith_round_votes.sum(axis = 0)
        num_candidates_left = np.sum(ith_vote_totals > 0)
        
        logger.info("\nSTV Round %d" % i)
        logger.info("# of winners found = %s", len(winner_list))
        logger.info("Candidate vote totals = %s" % ith_vote_totals)
        logger.info("Net for this round = %s" % np.sum(ith_vote_totals))
        logger.info("# of candidates left = %s", num_candidates_left)
        logger.debug('Voter data:\n%s' % data)
        round_results.append(ith_vote_totals)

        
        # Which candidates have won
        round_winners = np.where(ith_vote_totals >= quota)[0]
        
        if len(round_winners) > 0:
            
            # Retrieve ballots of winners. 
            for k in round_winners:
                if reallocation == 'hare':
                    surplus = ith_vote_totals[k] - quota
                    
                    winning_ballot_index = np.flatnonzero(data[:, k] == 1)
                    winning_data = data[winning_ballot_index]
                    winning_ballot_num = len(winning_data)
                    num2remove = winning_ballot_num - surplus
                    
                    shuffled_index = winning_ballot_index.copy()
                    rstate.shuffle(shuffled_index)
                    remove_index = shuffled_index[0 : num2remove]
                    retain_index = shuffled_index[num2remove :]
                else:
                    raise ValueError(reallocation + ' not valid reallocation method')
                    
                # Zero out winner votes that are not surplus.
                data[remove_index, :] = 0
                
                # Zero out the winner and re-order. 
                data[:, k] = 0
                
                # With reorder, 2nd choices can be sorted to 1st choice. 
                data = rcv_reorder(data)
                                  
                winner_list.append(k)
                
                logger.info("Winner Found = Candidate #%r" % k)
                logger.info("Surplus Votes = %d" % surplus)
                logger.debug("Winning ballots to remove = %s" % remove_index)
                logger.debug("Winning ballots transfer = %s" % retain_index)
                logger.info('')
                                    
        
        # begin candidate elimination
        else:
            if data.sum() == 0:
                logger.warning('Everyone has been elimated. No majority found... WARNING!')
            logger.info('No Winners Found. Start IRV Elimination')
            data, loser, ties = IRV_eliminate(data)
            if len(ties) > 1:
                logger.info("Ties found: %s", ties)                
                loser = ties[rstate.randint(0, 1)]
                logger.info("Picking a loser at random...")
                data[:, loser] = 0
                
            logger.info("Candidate %d eliminated." % loser)

    logger.info("WINNERS")
    logger.info(winner_list)
 
    return np.array(winner_list), ties, np.array(round_results)
        
                
 

def ____BROKEN_irv(data, numwin=1, num_eliminate=None):
    """
    Calculate winners of an election using Instant Runoff Voting
    
    Parameters
    ----------
    data : array shaped (a, b)
        Election voter rankings, from [1 to b].
        Data composed of candidate rankings, with
        
           - Voters as the rows, with `a` total voters
           - Candidates as the columns, with `b` total candidates.
           
        Use 0 to specify unranked (and therefore not to be counted) ballots.  
        
    numwin : int
        Number of winners
        
    Returns
    --------
    winners : array of shape (numwin,)
        Winning candidates index.
    ties : array of shape (tienum,)
        Tied candidates index for the last round, numbering 'tienum'.
    round_history : array of shape (numwin, b)
        Score summations for each candidate, for each round.        
    data : array of shape (a, b)
        Re-ordered ranking data after losers have been eliminated, retaining 
        winner ranksings.
    """
    data = np.array(data, copy=True)    
    candidate_num = data.shape[1]
    if num_eliminate is None:
        numrounds = max(1, candidate_num - numwin)
    else:
        numrounds = num_eliminate
        
    logger.info('# of rounds = %s', numrounds)
    logger.info('# of winners = %s', numwin)
    # initialize history datas
    round_history = []
    loserbool = np.ones(candidate_num, dtype=bool)
    
    
    for i in range(numrounds):
        data2 = data.copy()
        
        # Set of survivor candidates for each rond
        survived = set()
        num_left = candidate_num - (i + 1)
        
        # Second loop to check for ties. 
        for j in range(1, candidate_num+1):
            vote_index = (data2 == j)
            vote_count = np.sum(vote_index, axis=0)
            
            
            # Retrieve survivors of the round as winnersj. Add to survivors
            winnersj, tiesj = winner_check(vote_count, numwin=num_left)
            survived.update(set(winnersj))
            num_left = num_left - len(winnersj)
            round_history.append(vote_count)
            
            logger.info('\nRound %d, rank %d' % (i, j))
            logger.info('Counts = %s' % vote_count)
            logger.info('survivors=%s of %s' % (survived, num_left))
            
            # Break loop if no ties found and continue additional runoff rounds. 
            if len(tiesj) == 0:
                
                # Generate loser by inverting survivors
                loserbool[list(survived)] = False
                
                # Zero out loser rank data
                data[:, loserbool] = 0
                data = rcv_reorder(data)
                logger.info('losers=%s' % np.where(loserbool)[0])
                break
            else:
                # Zero out winner rank data temporarily for tie checking
                data2[:, winnersj] = 0

    ties = tiesj
    winners = np.sort(list(survived))
    round_history = np.array(round_history)
    logger.info('winners=%s', winners)
    logger.info('ties=%s', ties)    
    
    output = {}
    output['round_history'] = round_history
    output['data'] = data
    return winners, ties, output

   
#test_eliminate()
#    

#d = [[1, 2, 3],
#     [1, 3, 2],
#     [3, 2, 1],
#     [2, 3, 1],
#     [3, 1, 2]]
#

# Generate some random ballot results
