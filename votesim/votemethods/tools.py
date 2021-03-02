# -*- coding: utf-8 -*-

"""
Some common functions shared by many voting systems
"""
import numpy as np

def droop_quota(votes, seats):
    """
    Threshold of votes required to get elected for STV. 
    
    Parameters
    -----------
    votes : int
        Total number of votes cast in election
    seats : int
        Total number of seats needed to be filled.
    
    Returns
    -------
    out : int
        Droop quota; number of votes required to get elected. 
    """
    
    return int(np.floor(votes / (seats + 1)) + 1)


def hare_quota(votes, seats):
    return int(np.ceil(votes / seats))



def winner_check(results, numwin=1):
    """
    Check for winner and handle ties, considering results, where the greatest 
    result results in a win. 
    
    This function is used as a fundamental building block for other voting
    methods suchas plurality, score, STAR, etc. 
    
    Parameters
    ----------
    results : array of shape (a,)
        Result quantities for `a` # of candidates. Candidate index with
        the greatest result wins. 
        
        - Each element is the candidate "score" as int or float.
        - Set an element to np.NAN to ignore a particular candidate.
    numwin : int
        Number of winners to return
        
    Returns
    -------
    winners : array of shape(b,)
        Index locations of each winner. 
          - b = `numwin` if no ties detected 
          - b > 1 if ties are detected. 
    ties : array of shape (c,)
        Index locations of ties
        
        
    Example
    ---------
    We have run a plurality election for 4 candidates. The final tallies
    for each candidate are
    
    >>> counts = [2, 10, 4, 5]
    
    To determine the index location of the winner, use
    
    >>> w, t = winner_check(counts, numwin=1)
    """
    if numwin > 1:
        return _multi_winners_check(results, numwin=numwin)    
    
    # dummy, empty output array
    a = np.array([], dtype=int)
    sums = np.array(results)
    try:
        imax = np.nanargmax(sums)
    # Error occurs if all sums is NAN. If so, no winner, all ties.
    except ValueError:
        return a, np.arange(len(results))
        
    iscore = sums[imax]
    winners = np.where(sums == iscore)[0]
    wnum = len(winners)
    
    if wnum > 1:
        return a, winners
    elif wnum == 1:
        return winners, a
    else:
        raise RuntimeError('This state should not have been reached....')
        
        
    


def _multi_winners_check(results, numwin=1):
    """
    Multi-winner version of winner_check; see `winner_check` for arguments
    """
    results = np.array(results, copy=True, dtype='float64')
    
    # Get the ranking
    ranking = np.argsort(results)[::-1]
    
    # Filter out np.NAN 
    ikeep = ~np.isnan(results)
    ikeep = ikeep[ranking]    
    ranking = ranking[ikeep]
    
    winners = []
    ikeepnum = np.sum(ikeep)

    if numwin > ikeepnum:
        raise ValueError('Number of winners exceeds valid candidates of %s' % ikeepnum )
        
    # Loop through number of winners. 
    for j in range(numwin):
        winners_left = numwin - len(winners)
        candidate = ranking[j]
        cvotes = results[candidate]
        
        ties = np.where(cvotes == results)[0]
        tienum = len(ties)
        if tienum > winners_left:
            return np.array(winners, dtype=int), ties
        winners.append(candidate)
        
        # After winner has been added to list, set np.nan as his votes
        results[candidate] = np.nan
    
    ties = np.array([], dtype=int)
    winners = np.array(winners)
    return winners, ties

#
#def result_rankings(results):
#    """
#    Sort the results into winning ranks
#    """
#    
#    results = np.array(results, copy=True)
#    isort = np.argsort(results)[::-1]
#    results_unique = np.unique(results)
#    
#    for ru in results_unique:
#        


def winner_check_named(results, candidates: list, numwin: int=1):
    """Check winners variant for named candidates from a list. 
    
    Parameters
    ----------
    results : ndarray
        Result quantities for `a` # of candidates. Candidate index with
        the greatest result wins. 
        
        - Each element is the candidate "score" as int or float.
        - Set an element to np.NAN to ignore a particular candidate.
        
    candidates : ndarray
        Candidate names.
    numwin : int, optional
        Number of winners to return. The default is 1.

    Returns
    -------
    winners : array of shape(b,)
        Names of each winner
          - b = `numwin` if no ties detected 
          - b > 1 if ties are detected. 
    ties : array of shape (c,)
        Names of tie candidates
    """
    candidates = np.array(candidates)
    winners, ties = winner_check(results, numwin=numwin)
    win_candidates = candidates[winners]
    tie_candidates = candidates[ties]
    return win_candidates, tie_candidates    




def run_with_eliminated(
        func,
        eliminated: list, 
        data: np.ndarray, 
        numwin: int=1, 
        **kwargs):
    """Run election method with list of eliminated candidates.
    
    Parameters
    ----------
    func : function
        Election method
    eliminated: list[int]
        List of eliminated candidate indices
    data : ndarray (a, b)
        Ballot data for `a` voters and `b` candidates
    numwin : int
        Number of winners to find. 
        
    Returns
    -------
    winners : ndarray (c,)
        Winner candidate indices
    ties : ndarray (t,)
        Tie candidates indices
    outputs : dict
        data output dict
    """
    num_candidates = data.shape[1]
    
    cbools = np.ones(num_candidates, dtype=bool)
    cbools[eliminated] = False
    cindex = np.where(cbools)[0]
    
    data1 = data[:, cindex]
    winners1, ties1, output1 = func(data1, numwin=numwin, **kwargs)
    
    winners = cindex[winners1]
    ties = cindex[ties1]
    return winners, ties, output1
    



def multi_win_eliminate(func, data, numwin=1, **kwargs):
    """Convert single winner method to multi-winner method, 
    using candidate elimination."""        
    winners = []
    num_left = numwin
    data = data.copy()
    outputs = []
    ties = None
    while num_left > 0:
        
        winners_ii, ties_ii, output_ii = run_with_eliminated(
            func,
            winners, 
            data,
            numwin=1,
            **kwargs)
        
        outputs.append(output_ii)
        winner_len = len(winners_ii)
        tie_len = len(ties_ii)
        
        # Check if single winner has been found
        if winner_len == 1:
            winners_ii = winners_ii[0]
            winners.append(winners_ii)
        
        elif winner_len == 0:
            # Check if ties have been found, but there are enough 
            # slots left to fill them in as winners.
            if num_left >= tie_len:
                winners.extend(ties_ii)
                
            # Check for ties, but there are no slots left for them all. 
            elif tie_len > 1:
                ties = ties_ii
                break
        
        num_left = numwin - len(winners)
    
    winners = np.array(winners, dtype=int)
    if ties:
        ties = ties.asarray(dtype=int)
    else:        
        ties = np.array([], dtype=int)
    return winners, ties, outputs
                

def multi_win_eliminate_decorator(func):
    """Decorator to convert single winner method to multi-winner method, 
    using candidate elimination.""" 

    def func2(data, numwin=1, **kwargs):    
        if numwin == 1:
            return func(data, numwin=numwin, **kwargs)
        
        winners, ties, out1 = multi_win_eliminate(
            func, 
            data,
            numwin = numwin,
            **kwargs
        )    
        output = out1[0].copy()
        output['elimination_rounds'] = out1       
        return winners, ties, output
    
    return func2
            
            
    

    
def rcv_reorder(data):
    """Make sure rankings are sequential integers from [1 to b],
    with 0 meaning eliminated or unranked.
    
    Parameters
    ----------
    data : array shaped (a, b)
        Election voter rankings, from [1 to b].
        Data composed of candidate rankings, with
        
           - Voters as the rows
           - Candidates as the columns. 
           
    Returns
    --------
    out : array shaped (a,b)
        Conversion of scores to integer ranked sequenced data from [1 to b].
    """
    data = np.copy(data)
    max_rank = np.max(data)
    unranked = (data == 0)
    data[unranked] = max_rank + 10
    
    # Need to argsort twice to sort and then unsort but retain integer increments.
    data = np.argsort(np.argsort(data, axis=1), axis=1) + 1
    data[unranked] = 0
    return data
    

def score2rank(data, cutoff=None):
    """
    Convert scores or ratings to rankings starting from 1.
    
    Parameters
    -----------
    data : array shaped (a, b)
        Election voter rating, 0 to max. 
        Data composed of candidate ratings, with
        
           - `a`-number of Voters as the rows
           - `b`-number of Candidates as the columns. 
           
        Use 0 to specify unranked (and therefore not to be counted) ballots.  
        
         - a : number of voters dimension.
         - b : number of candidates. A score is assigned for each
               candidate from 0 to b-1.
    cutoff : None or float
        If float, specify a cutoff rating where ratings below this value are
        unranked as ranking=0.
        
        
    Returns
    -------
    rank : array shaped (a, b)
        Election rankings of each voter for each candidate
        
        - a : voters dimension
        - b : candidates dimension
    
    """
    data = np.array(data)
    izero = data <= 0
    
    i = np.argsort(np.argsort(-data, axis=1), axis=1) + 1
    i[izero] = 0
    
    if cutoff is not None:
        ic = data <= cutoff
        i[ic] = 0
    
    return i



def getplurality(ratings=None, ranks=None):
    """
    Retrieve first-choice plurality votes, given either ratings or rankings data
    
    Parameters
    ----------
    ratings : array shape (a, b)
        Ratings data. Mutually exlusive with ranks
    ranks : array shape (a, b)
        Rankings data. Mutually exclusive with ratings
        
    Returns
    --------
    out : array shape (a, b)
        Rank/ratings data converted into single-vote plurality ballots.
    
    
    """
    # convert possible cardinal data to a single choice. 
    if ranks is not None:
        return (ranks==1).astype(int)
    elif ratings is not None:
        data = np.atleast_2d(ratings)
        new = np.zeros(data.shape)
        i0 = np.arange(data.shape[0])
        i1 = np.argmax(data, axis=1)
        
        index_blanks = data == 0
#        logger.debug('i0 = %s', i0)
#        logger.debug('i1 = %s', i1)
        new[i0, i1] = 1
        new[index_blanks] = 0
    
        return new


def handle_ties(winners, ties, numwinners, rstate=None):
    """If ties are found, choose random tied candidate to break tie
    
    Parameters
    ----------
    winners : array shaped (a,)
        Winning candidate indices
    ties : array shaped (b,)
        Candidates that almost won but have received a tie with other candidates. 
    numwinners : int
        Total number of required winners for this election
    
    Returns
    --------
    winner : array shaped (numwinners,)
        Winners of election. Tie-broken by random choice. 
    """
    assert len(winners) + len(ties) > 0, 'Empty winner and ties input.'
    
    if rstate is None:
        rstate = np.random.RandomState
    
    winners = np.array(winners)
    num_found = len(winners)
    num_needed =  numwinners - num_found
    
    if num_needed > 0:
        
        new = rstate.choice(ties, size=num_needed, replace=False)
        winners = np.append(winners, new)
    return winners.astype(int)



