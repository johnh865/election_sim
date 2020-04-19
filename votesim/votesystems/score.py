
import numpy as np

import logging
from votesim.votesystems import tools
logger = logging.getLogger(__name__)




def _set_maxscore(data, scoremax):
    """Set the max score discretization"""
    dmax = np.max(data)
    data = data / dmax
    data = np.round(data * scoremax)
    return data


def score(data, numwin=1):
    """
    Score voting.
    
    Parameters
    -----------
    data : array shaped (a, b)
        Election voter scores, 0 to max. 
        Data of candidate ratings for each voter, with
        
           - `a` Voters represented as each rows
           - `b` Candidates represented as each column. 
              
    numwin : int
        Number of winners to consider
    
    Returns
    -------
    winners : array of shape (numwin,)
        Winning candidates index.
    ties : array of shape (tienum,)
        Tied candidates index for the last round, numbering 'tienum'.
    sums : array of shape (numwin, b)
        Score summations for each candidate.
    """
    data = np.atleast_2d(data)
    sums = np.sum(data, axis=0)
    winners, ties = tools.winner_check(sums, numwin=numwin)
    return winners, ties, sums


def score5(data, numwin=1):
    data = _set_maxscore(data, 5)
    return score(data, numwin=numwin)


def score10(data, numwin=1):
    data = _set_maxscore(data, 10)
    return score(data, numwin=numwin)


def majority_judgment(data, numwin=1, remove_nulls=True):
    """
    Majority judgment (median score).
    
    Parameters
    -----------
    data : array shaped (a, b)
        Election voter scores, 0 to max. 
        Data of candidate ratings for each voter, with
        
           - `a` Voters represented as each rows
           - `b` Candidates represented as each column. 
              
    numwin : int
        Number of winners to consider
        
    remove_nulls : bool
        If True (default), remove any ballots that are marked with 
        all zeros. 
    
    Returns
    -------
    winners : array of shape (numwin,)
        Winning candidates index.
    ties : array of shape (tienum,)
        Tied candidates index for the last round, numbering 'tienum'.
    sums : array of shape (numwin, b)
        Median scores for each candidate.
    """    
    
    data = np.atleast_2d(data)
    if remove_nulls:
        index = np.all(data == 0, axis=1)
        data = data[~index]
        
    medians = np.median(data, axis=0)
    winners, ties = tools.winner_check(medians, numwin=numwin)
    return winners, ties, medians




def reweighted_range(data, numwin=1, C_ratio=1.0, maxscore=None):
    """
    Multi-winner election using reweighted range voting.
    
    https://www.rangevoting.org/RRVr.html


    Parameters
    -----------
    data : array shaped (a, b)
        Election voter scores, 0 to max. 
        Data of candidate ratings for each voter, with
        
           - `a` Voters represented as each rows
           - `b` Candidates represented as each column. 
              
    numwin : int
        Number of winners to consider
    
    C_ratio : float
        Proportionality factor
        
        - C_ratio = 1.0 -- M; Greatest divisors (d'Hondt, Jefferson) proportionality
        - C_ratio = 0.5 -- M/2; Major fractions (Webster, Saint-Lague) method
        
    maxscore : None (default), float
        Maximum score to use for calculation of C. Use max of data if None.
        
    Returns
    -------
    winners : array of shape (numwin,)
        Winning candidates index.
    ties : array of shape (tienum,)
        Tied candidates index for the last round, numbering 'tienum'.
    round_history : array of shape (numwin, b)
        Score summations for each candidate, for each round.
        
        - rows *numwin* -- Represents each round for total number of winners
        - columns *b* -- Represents each candidate. 
        - data is net score of each candidate for each round.     
    """
    data = np.array(data)
    data = np.atleast_2d(data)
    
    if maxscore is None:
        maxscore = np.max(data)
        
    ballot_num = data.shape[0]    
    C = maxscore * C_ratio
    
    # Set initial weights as uniform
    weights = np.ones(ballot_num)
    
    winners = []        # Store winning candidate indices here.
    ties = []           # Store tie candidagte indices here.
    round_history = []  # Store the history of net scores for each round. 
    
    winner_sum = np.zeros(ballot_num)                # Store the total score given to winners by each voter
                            
    
    for i in range(numwin):                          # Loop through for number of winners. 
        
        data_weighted = data * weights[:, None]        
        sums = np.sum(data_weighted, axis=0)         # Calculate weighted net score for each candidate
        winnersi, tiesi = tools.winner_check(sums)
        if len(winnersi) == 0:
            winner = tiesi[0]
        else:
            winner = winnersi[0]
            
#        winnersi = score_winners_check(sums)         # Get candidate with greatest score. If tie, return multiple candidates. 
#        winner = winnersi[0]
        winner_sum = winner_sum + data[:, winner]    # Calculate total winning scores from  each voter
        weights = C / (C + winner_sum)               # Calculate new weighting
        
        logger.debug('scores = ')
        logger.debug(data)
        logger.debug('weights = ')
        logger.debug(weights)

        # Handle ties for last winner
        if len(winnersi) > 1:
            # attempt to break tie using majoritarianism./?? NOT IMPLEMENTED YET....
            
            ties = winnersi
        else:
            winners.append(winner)            
            
        logger.info('\nRound #%d' % i)
        logger.info('net scores = %s' % sums)
        logger.info('round winner = %s' % winner)        

        data[:, winner] = 0
        round_history.append(sums)
        
    winners = np.array(winners)
    logger.info('winners = %s' % winners)
    logger.info('ties = %s' % ties)
    return winners, ties, np.array(round_history)




def star(data, numwin=1):
    """
    STAR voting (Score then Automatic Runoff)
    
    Parameters
    ----------
    data : array shaped (a, b)
        Election voter scores, 0 to max. 
        Data of candidate ratings for each voter, with
        
           - `a` Voters represented as each rows
           - `b` Candidates represented as each column. 
    numwwin : int
        Multi-winners... parameter > 1 not supported!!
        
    """       
    if numwin > 1:
        raise NotImplementedError('Multi-winner STAR not available.' )
    ### First Round -- Score Voting
    data = np.array(data)
    sums = np.sum(data, axis=0)
    
    ### Retrieve Runoff Winners
    winners, ties = tools.winner_check(sums, numwin=2)
    runoff_candidates = np.append(winners, ties)
    runoff_data = data[:, runoff_candidates]
    
    ### Second Round -- Automatic majoritarian runoff
    # The candidate that beats the most head-to-head competitions wins!
    vote_matrix = []
    
    # Calculate number of positive votes for candidate head to head
    for icandidate in runoff_candidates:
        iscores = data[:, icandidate : (icandidate+1)]
        votes_for = (iscores > runoff_data).sum(axis=0)
        votes_against = (iscores < runoff_data).sum(axis=0)
        votes = votes_for - votes_against
        vote_matrix.append(votes)
    
    vote_matrix = np.array(vote_matrix)
    win_matrix = vote_matrix > 0
    win_array = np.sum(win_matrix, axis=1)
    
    # Calculate winner
    jwinners, jties = tools.winner_check(win_array, numwin=1)
    winners = runoff_candidates[jwinners]
    ties = runoff_candidates[jties]
    
    details = {}
    details['sums'] = sums
    details['runoff_candidates'] = runoff_candidates
    details['runoff_matrix'] = vote_matrix
    details['runoff_sums'] = win_array
    
    return winners, ties, details



def star5(data, numwin=1):
    data = _set_maxscore(data, 5)
    return star(data, numwin=numwin)


def star10(data, numwin=1):
    data = _set_maxscore(data, 10)
    return star(data, numwin=numwin)



def approval50(data, numwin=1):
    
    dmax = np.max(data)
    data = np.round(data / dmax)
    return score(data, numwin=numwin )


def approval100(data, numwin=1, threshold=.01):
    dmax = np.max(data)
    data = data / dmax
    data = (data > threshold) * 1.0
    return score(data, numwin=numwin)


def approval75(data, numwin=1):
    return approval100(data, numwin=numwin, threshold=.25)
    
    
    

def sequential_monroe(data, numwin=1, maxscore=None ):
    """
    Multi-winner score based on Parker_Friedland's Reddit post
    https://www.reddit.com/r/EndFPTP/comments/auyxny/can_anyone_give_a_summary_of_multiwinner_methods/ehgkfbl/
    

    1. For candidate X, sort the ballots in order of highest score given to candidate X to lowest score given to candidate X.

    2. Calculate the average score given to X on the first hare quota of those ballots. Record this score as that candidate's hare quota score. See Footnote.

    3. Repeat this process for every candidate.

    4. Elect the candidate with the highest hare quota score and exhaust the votes that contribute to that candidate's hare quota score.

    5. Repeat this process until all the seats are filled.
    
    Parameters
    ---------
    data : array shaped (a, b)
        Election voter scores, 0 to max. 
        Data of candidate ratings for each voter, with
        
           - `a` Voters represented as each row.
           - `b` Candidates represented as each column. 
              
    numwin : int
        Number of winners to consider           
        
    Returns
    -------
    winners : array of shape (numwin,)
        Winning candidates index.
    ties : array of shape (tienum,)
        Tied candidates index for the last round, numbering 'tienum'.
    round_history : array of shape (numwin, b)
        Average scores of top quota for each candidate, for each round.
        
        - rows *numwin* -- Represents each round for total number of winners
        - columns *b* -- Represents each candidate. 
        - data is net score of each candidate for each round.        
    
    """
    data = np.array(data)
    num_candidates = data.shape[1]
    num_voters = data.shape[0]    
    quota = tools.hare_quota(num_voters, numwin)
    
    winners = []
    mean_scores_record = []
    
    # Find for each required number of winners
    for i in range(numwin):
        mean_scores = []
        top_voter_record = []
        
        for ic in range(num_candidates):
            # sort candidate's scores highest to lowest, get top quota voters
            candidate_top_voters = np.argsort(data[:, ic])[::-1][0 : quota]
            candidate_top_scores = data[candidate_top_voters, ic]
            mean_score = np.mean(candidate_top_scores)
            mean_scores.append(mean_score)
            top_voter_record.append(candidate_top_voters)
            
        mean_scores = np.array(mean_scores)
        mean_scores_record.append(mean_scores)
        
        # Get winner from mean_scores, check for ties. 
        winner, ties = tools.winner_check(mean_scores, 1)
        if (len(ties) > 1) and (i == numwin - 1):
            #HANDLE TIES ONLY AT LAST WINNER
            return np.array(winners), ties, np.array(mean_scores_record)
        else:
            winner = winner[0]
            winners.append(winner)
        
        # zero out the ballots of the winner's voters. 
        exhausted_ballots = top_voter_record[winner]
        data[exhausted_ballots, :] = 0
        
    winners = np.array(winners, dtype=int)
    ties = np.array([], dtype=int)
    return winners, ties, np.array(mean_scores_record)
        
          


# Generate some random ballot results
if __name__ == '__main__':
    d = [[10, 9, 8, 1, 0]] * 60 + [[0, 0, 0, 10, 10]] * 40
    #d  = [[10, 10, 10, 1]] * 5
    d = np.array(d)
    
    # Call the STV function
    w = reweighted_range(d, 3)
    
    
        
            