# -*- coding: utf-8 -*-

"""
Simulate voting behavior. 
Convert voter preferences into preferences & rankings for candidates.

Typical function inputs
------------------------
voters : array shape (a, n)
    Voter preferences; a-dimensional voter preferences for n issues. 
    
    - Each row of the array represents a voter's preferences
    - Each column represents a difference issue.
    
candidates : array shape (b, n)
    Candidate preferences for n-dimensional issues. 
    
    - Each row represents a candidate's preference
    - Each column represents a difference issue. 
    
    
Types of voting behavior to simulate
-----------------------------------

Preference Tolerance
====================
Assume that voters have a threshold of preference distance, at which if a
candidate exceeds the tolerance threshold, they will give zero score or
refuse to rank the candidate. 


Preference Tolerance -- Last Hope
=================================
Assume that for some voters, if all candidates exceed their threshold of 
preference tolerance, they might decide still vote for the single closet candidate.


Memory limitation
==================
Assume that voters might only rate/rank so many candidates, for example,
up to a limit of about 7.

Preference Error
================
Voters might miscalculate their preference distance from a candidate.




"""
import numpy as np
import logging
logger = logging.getLogger(__name__)



def voter_distances(voters, candidates, weights=None, order=1):
    """
    Calculate preference distance of voters away from candidates.
    
    Parameters
    ----------
    voters : array shape (a, n)
        Voter preferences; `a` number of voters, cardinal preferences for `n` issues. 
    candidates : array shape (b, n)
        `b`-number of Candidate preferences for `n`-dimensional issues.     
    weights : None or array shape (a, n)
        Dimensional weightings of each voter for each dimension.
        Only relevant if n > 1
        
        
    order : int
        Order of norm
        
        * 1 = taxi-cab norm; preferences for each issue add up
        * 2 = euclidean norm; take the sqrt of squares. 
        
        
    Returns
    -------
    distances : array shaped (a, b)
        Candidate `b` preference distance away from voter, for each voter `a`
    """
    
    # diff shape = (voternum, n, 1) - (1, candnum, n)
    # diff shape => (voternum, candnum, n)
    diff = voters[:, None] - candidates[None, :]
        
    if diff.ndim == 2:
        distances = np.abs(diff) 
    elif diff.ndim == 3:
        # Apply weights to each candidate via loop
        if weights is not None:
            for ii in range(voters.shape[1]):
                diff = diff[:, :, ii] * weights[:, ii]        
            
        distances = np.linalg.norm(diff, ord=order, axis=2)  
    else:
        s = 'Wrong number of dimensions input for voters %s or candidate %s'
        s = s % (voters.shape, candidates.shape)
        raise ValueError(s)
    return distances



def voter_distance_error(distances, error_std, rstate=None):
    """
    Add error to voter distances from candidates
    
    Parameters
    ----------
    distances : array shape (a,b)
        Distances generated by function `voter_distances`
    error_std : array shape (a,)
        standard deviation of distance error for each voter.
        Each voter has an accuracy represented by this. 
        
    Returns
    ---------
    distances : array shape (a,b)
        New voter distances including voter error. 
        
    """
    if rstate is None:
        rstate = np.random.RandomState
    error_std = np.atleast_1d(error_std)
    enorm = rstate.normal(0.0, scale=1.0, size=distances.shape)
    error = enorm * error_std[:, None]
    return distances + error
    
    
            
            
def voter_rankings(voters, candidates, cnum=None, _distances=None):
    """
    Create rankings of voter for candidates, by considering only
    a top `cnum` of candidates and ignoring the rest. The last candidate 
    is rated 0. The closest candidate is rated 1. Others are linearly scaled in between. 
    
    Parameters
    ----------
    voters : array shape (a, n)
        Voter preferences; `a` number of voter cardinal preferences for `n` issues. 
    candidates : array shape (b, n)
        Candidate preferences for n-dimensional issues. 
    cnum : None (default), int, or int array shaped (a,)
        Max number of candidates that will be ranked. Can be set for each voter.        

    Returns
    ------
    rankings : array shaped (a, b)
        Voter rankings for each candidate
    """
    

    
    # Create preference differences for every issue. 
    # diff = shape of (a, n, b) or (a, b)
    # distances = shape of (a, b)
    if _distances is not None:
        distances = _distances
        vnum = len(distances)
    else:
        vnum = len(voters)
        distances = voter_distances(voters, candidates)
        
    # ranking of candidate preferences
    i_rank = np.argsort(distances, axis=1) + 1
    
    if cnum is None:
        return i_rank
    else:
        cnum = np.array(cnum)
        if cnum.size == 1:
            cnum = np.ones(vnum) * cnum
    
    logger.debug('i_rank =\n %s', i_rank)
    logger.debug('cnum = %s', cnum)
    
    # get ranks to zero out
    i_kill = (i_rank > cnum[:, None])
    i_rank[i_kill] = 0
    return i_rank



def voter_scores_by_tolerance(voters, candidates, 
                              distances=None,
                              weights=None,
                              tol=1, cnum=None, error_std=0, strategy='abs',):
    """
    Creating ratings from 0 to 1 of voter for candidates 
    using a tolerance circle.
    
    - Assume that all voters are honest.
    - Assume that voters have a preference "tolerance". 
      Candidates whose preference distance exceeds this tolerance have utility
      set to zero. 
      - Linear mapping of preference distance and utility. 
      - Utility = 1 if candidate preference is exactly voter preference.
      - Utility = 0 if candidate preference is exactly the tolerance distance. 
    - Assume that voters will give strongest possible preference to closest candidate,
      unless that candidate is outside their preference tolerance. 
    
    Parameters
    ----------
    voters : array shape (a, n)
        Voter preferences; `a` voter cardinal preferences for `n` issues. 
    candidates : array shape (b, n)
        `b` number of candidate preferences for `n`-dimensional issues. 
    tol : float, or array shaped (a,)
        Voter candidate tolerance distance. If cardinal preference exceed tol, 
        utility set to zero. Toleranace is in same units as voters & candidates
    cnum : None (default), int, or int array shaped (a,)
        Max number of candidates that will be ranked. Can be set for each voter.        
        
    strategy : str
        Tolerance determination strategy. Use the following string options:
        
        - "abs" -- set `tol` as an absolute value to compare to distance.
        - "voter" -- set `tol` as factor of the average voter distance from the centroid.
          - At tol=1, candidates farther than 1 x avg voter distance are rated zero
          - At tol=2, candidates farther than 2 x avg voter distance are rated zero.
        - "candidate" -- set `tol` relative to farthest distance of candidate to voter. 
          - At tol=1, the max distance candidate is rated zero.
          - At tol=0.5, candidates at half of the max distance are rated zero
          
    Returns
    -------
    out : array shaped (a, b) 
        Utility scores from a-voters for each b-candidate. 
    """
    
    # Create preference differences for every issue. 
    # diff = shape of (a, n, b) or (a, b)
    # distances = shape of (a, b)
    if distances is None:
        distances = voter_distances(voters, candidates, weights=weights)
#        distances = voter_distance_error(distances, error_std, rstate=rstate)
    
    if strategy == 'abs':
        dtol = tol
    elif strategy == 'voter':
        v_mean = np.mean(voters, axis=0)
        v_dist = voter_distances(voters, v_mean[None, :], weights=weights)
        dmean = np.mean(v_dist)
        dtol = dmean * tol
        
    elif strategy == 'candidate':
        dmax = np.max(distances, axis=1)
        dtol = dmax * tol
    else:
        raise ValueError('Incorrect strategy arg = %s' % strategy)
        
    logger.info('strategy=%s' % strategy)
    logger.info('relative tol = %s', tol)
#    logger.info('absolute dtol = %s', dtol)
    
    
    dtol = np.array(dtol)
    if dtol.ndim == 1:
        dtol = dtol[:, None]
    
    # Get utility from tol
    i = np.where(distances > dtol)
    utility = (dtol - distances) / dtol
    utility[i] = 0
    
    # Set to zero candidates past max number to score
    if cnum is not None:
        ranks = voter_rankings(voters, candidates,
                               cnum=cnum,
                               _distances=distances)
        iremove = ranks <= 0
        utility[iremove] = 0
    
    #rescale utility so favorite has max score,
    max_utility = np.max(utility, axis=1)    
    min_utility = np.min(utility, axis=1)
    
    
    i2 = np.where(max_utility == 0)
    max_utility[i2] = 1.  # avoid divide by zero error for unfilled ballots
    
    
    umax = max_utility[:, None]
    umin = min_utility[:, None]
    
    # Rescale utilities so that least favorite has zero score. 
    ratings =  utility / max_utility[:, None]
    ratings = (utility - umin) / (umax - umin)
    ratings = np.maximum(0, ratings)
    return ratings
    

def voter_scores_by_rank(voters, candidates, cnum=None):
    """
    Calculate scores by assuming voters will only rate a set number `cnum` 
    of candidates from 0 to 1. Tolerance is based on number of candidates 
    rather than preference distance. 
    
    
    Parameters
    ----------
    voters : array shape (a, n)
        Voter preferences; n-dimensional voter cardinal preferences for n issues. 
    candidates : array shape (b, n)
        Candidate preferences for n-dimensional issues. 
    cnum : None (default), int, or int array shaped (a,)
        Max number of candidates that will be ranked. Can be set for each voter.
    
    Returns
    -------
    out : array shaped (a, b) 
        Utility scores from a-voters for each b-candidate. 
    """
    
    
    distances = voter_distances(voters, candidates)
    rankings = voter_rankings(voters, candidates, cnum, distances)
    
    iremove = rankings <= 0
    distances[iremove] = -1
    dtol = np.max(distances, axis=1)

    utility = (dtol - distances) / dtol
    utility[iremove] = 0
    
    #rescale utility so favorite has max score
    max_utility = np.max(utility, axis=1)    
    i2 = np.where(max_utility == 0)
    max_utility[i2] = 1.
    return utility / max_utility[:, None]    


def __voter_scores_log(voters, candidates, distances=None, weights=None):
    
    raise NotImplementedError()
    
    if distances is None:
        distances = voter_distances(voters, candidates, weights=weights)
        
    utility = -distances    
    U = utility
        
    #rescale utility so favorite has max score,
    max_utility = np.max(U, axis=1)    
    min_utility = np.min(U, axis=1)    

    # M is max score
    M = 1
    
    def cap(x):
        return np.minimum(M, np.maximum(0, x))
        
    umax = max_utility[:, None]
    umin = min_utility[:, None]
    R = (umin / umax) ** (1/M)
    
    S = cap(M - np.log(U / umax) / np.log(R))
    return S
    



def __voter_scores_log(voters, candidates, distances=None, weights=None):
    raise NotImplementedError()
    if distances is None:
        distances = voter_distances(voters, candidates, weights=weights)
        
    utility = -distances

        
    #rescale utility so favorite has max score,
    max_utility = np.max(utility, axis=1)    
    min_utility = np.min(utility, axis=1)    
#    i2 = np.where(max_utility == 0)
#    max_utility[i2] = 1.  # avoid divide by zero error for unfilled ballots
    
    # Smax is max score
    Smax = 1
    m = Smax / np.log(max_utility / min_utility)
    S0 = -m * np.log(min_utility)
    m = m[:, None]
    S0 = S0[:, None]
    
    
    S = m * np.log(utility) + S0
    
    return S




    


def zero_out_random(ratings, limits, weights=None, rs=None):
    """
    Zero-out scores or ranks by random. Simluation of limits of voter information,
    where for many candidates, voters may not be informed of all of them.
    
    Parameters
    ----------------
    ratings : array shaped (a, b) 
        Ratings, scores, ranks from a-voters for each b-candidate. 
    limits : int array shaped (a,)
        Number of canidates each voter has knowledge about
    weights : array shaped (b,)
        Media model, probability weighting for each candidate where some 
        candidates are more likely to be known than others. 

    Returns
    -------
    out : array shaped (a,b)
        Adjusted ratings, scores, ranks with candidate limits applied. 
        
    """    
    ratings = np.copy(ratings)
    limits = limits.astype(int)
    vnum, cnum = ratings.shape
    remove_arr = np.maximum(0, cnum - limits)    
    
    if rs is None:
        rs = np.random.RandomState()
    
    for i, remove_num in enumerate(remove_arr):
        index = rs.choice(cnum, size=remove_num, p=weights, replace=False)
        ratings[i, index] = 0
    return ratings
        
    

    
    


def __zero_out_random(ratings, climits, weights=None, rs=None):
    """
    Zero-out scores or ranks by random. Simluation of limits of voter information,
    where for many candidates, voters may not be informed of all of them.
    
    Parameters
    ----------------
    scores : array shaped (a, b) 
        Ratings, scores, ranks from a-voters for each b-candidate. 
    climits : int array shaped (a,)
        Number of canidates each voter has knowledge about
    weights : array shaped (b,)
        Media model, probability weighting for each candidate where some 
        candidates are more likely to be known than others. 

    Returns
    -------
    out : array shaped (a,b)
        Adjusted ratings, scores, ranks with candidate limits applied. 
        
    """
    raise NotImplementedError()
    climits = climits.astype(int)
    if rs is None:
        rs = np.random.RandomState()

    ratings = np.atleast_2d(ratings).copy()
    voter_num, candidate_num = ratings.shape
    if weights is None:
        weights = np.ones(candidate_num)    
    wsum = np.sum(weights)
    weights = weights / wsum
#    premove = 1 - weights
#    premove = premove / np.sum(premove)
#    
#    cremoves  = candidate_num - climits
#    cremoves[cremoves < 0] = 0
    
    iremove0 = np.ones(candidate_num, dtype=bool)
    
    for i, clim in enumerate(climits):
        iremove = iremove0.copy()
        index = rs.choice(candidate_num, size=clim, p=weights, replace=False)
        iremove[index] = False
        ratings[i, iremove] = 0 
    return ratings



    
    


