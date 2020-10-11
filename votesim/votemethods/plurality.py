# -*- coding: utf-8 -*-
"""Plurality voting implementation."""

import numpy as np
import logging

from votesim.votemethods import tools
logger = logging.getLogger(__name__)
__all__ = ['plurality']


def plurality(data, numwin=1):
    """Run plurality election.
    
    Parameters
    ----------
    data : array shape (a, b)
        Election scoring data, 0 to 1. If rating data is input, plurality will find 
        maximum rated candidate. 
    
    numwin : int
        Number of winners. For numwin > 1, plurality turns into Single No Transferable
        Vote multi-winner election.
        
        
    Returns
    -------
    winners : array shape (numwin,)
        Winning candidate indices
    ties: array shaped(numties,)
        If there are tied candidates, return candidate indices here.
        If no ties, return empty array
    results : array shaped(b,)
        End vote count    
    """
    new = tools.getplurality(ratings=data)
    sums = np.sum(new, axis=0)
    winners, ties = tools.winner_check(sums, numwin=numwin)
    
    output = {}
    output['tally'] = sums
    return winners, ties, output



def plurality1(data, numwin=1):
    """Run plurality election.
    
    Parameters
    ----------
    data : array shape (a, b)
        Election scoring data, 0 to 1. If rating data is input, plurality will find 
        maximum rated candidate. 
    
    numwin : int
        Number of winners. For numwin > 1, plurality turns into Single No Transferable
        Vote multi-winner election.
        
        
    Returns
    -------
    winners : array shape (numwin,)
        Winning candidate indices
    ties: array shaped(numties,)
        If there are tied candidates, return candidate indices here.
        If no ties, return empty array
    results : array shaped(b,)
        End vote count    
    """
    # Variable descriptions
    # sums : array shape (b,)
    #   Vote totals for all candidates
    
    # convert possible cardinal data to a single choice. 
    new = tools.getplurality(ratings=data)
    
    # Get the winner
    logger.debug('vote data new:')
    logger.debug('\n%s' % new)
    sums = np.sum(new, axis=0)
    
    sums_out = sums.copy().astype(int)
    logger.info('vote totals:')
    logger.info(sums_out)
    ranking = np.argsort(sums)[::-1]
    
    logger.info('candidate # sorted list (first with most votes, last with least:')
    logger.info(ranking)
#    smax = sums[ranking[numwin - 1]]

    # check for tie condition
#    ties = np.where(sums == smax)[0]    
#    tienum = len(ties)
#    logger.info('%s ties found' % tienum)
    
    winners = []
    
    # Loop through number of winners. 
    for j in range(numwin):
        winners_left = numwin - len(winners)
        candidate = ranking[j]
        cvotes = sums[candidate]
        
        logger.info('candidate #%d' % candidate)
        logger.info(' - number of votes = %d' % cvotes)
        
        ties = np.where(cvotes == sums)[0]
        tienum = len(ties)
        if tienum > winners_left:
            logger.info(' - tie detected for winner slot #%d out of %d' % (j, numwin))
            logger.info(' - tie candidates = %s', ties)
            logger.info('winners=%s', winners)
            
            return winners, ties, sums
        logger.info(' - winner detected for candidate %d' % candidate)
        winners.append(candidate)
        
        # After winner has been added to list, zero out his votes
        sums[candidate] = 0
        
        
    logger.info('winners=%s', winners)
    return winners, np.array([]), sums_out
        
            
        

