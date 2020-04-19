# -*- coding: utf-8 -*-
"""
Various metrics to measure performance of an election
"""
import logging


import numpy as np
import scipy
import scipy.interpolate
from scipy.interpolate import NearestNDInterpolator

from votesim.models import vcalcs
from votesim import utilities
logger = logging.getLogger(__name__)


def bayesian_regret(voters, winner):
    """
    Calculate bayesian regret 
    
    
    voters : array (a, n)
        Voter preferences; n-dimensional voter cardinal preferences for n issues. 
    winner : array shape (n)
        Winner preferences for winner and `n`-dimensional issues.     
    """
    
    voters = np.atleast_2d(voters)
    num = len(voters)
    winner = np.atleast_2d(winner)
    
    centroid = np.mean(voters, axis=0)
    
    
    dw = np.sum(np.linalg.norm(voters - winner)) / num
    dc = np.sum(np.linalg.norm(voters - centroid)) / num
    return dc - dw




#def interp_nearest(x, y):
#    x = np.array(x)
#    if x.shape[1] == 1:
    

class PrRegret(object):
    """
    Measure "proportional representation" regret for multiwinner elections.
    
    - Calculate preference distance of each voter from his nearest-neighbor winning 
    representative.
    - Sum up distances to estimate the net voter "PR Regret" 
    
    Parameters
    -----------
    voters : array, shape (a, n)
        Voter preferences; n-dimensional voter cardinal preferences for n issues. 
    winners : array, shape (b, n)
        Winner preferences for `b` winners and `n`-dimensional issues.
    """
    
    def __init__(self, voters, winners, weights=None):
        self.voters = np.atleast_2d(voters)
        self.winners = np.atleast_2d(winners)       
        self.weights = weights
        
        self._num_winners = len(self.winners)
        self._num_voters, self._ndim = self.voters.shape
        return
    
    
    @utilities.decorators.lazy_property
    def nearest_winners(self):
        """array shape (a,)
            index locations of the nearest winners for each voter, 
            `a` total voters
        """
        winners = self.winners
        voters = self.voters        
        num_winners = self._num_winners
        
        # Create winner's array index
        winner_ids = np.arange(num_winners)
        
        # build function `interp` to obtain a voter's nearest winner, who is his representative
        
        # interp : function
        #    Take voter preference as input and output the nearest winner. 
        
        # voter_reps : array shaped (a)
        #    Index location of nearest candidate for `a` voters
        
        # if only one winner input
        if winners.shape[0] == 1:
            voter_reps = np.zeros(self._num_voters)
            
        # if multiple winners and N-dimensional
        elif voters.shape[1] > 1:
            interp = NearestNDInterpolator(winners, winner_ids)
            
            # Retrieve the voter's nearest representative
            voter_reps = interp(voters)
            
        # if multiple winners and 1-dimensional
        elif voters.shape[1] == 1:
            w = winners[:, 0]
            v = voters[:, 0]
            interp = scipy.interpolate.interp1d(w, winner_ids, 
                                                kind='nearest',
                                                fill_value='extrapolate')
            voter_reps = interp(v)
            
        voter_reps = voter_reps.astype(int)
        return voter_reps
    
    
    @utilities.decorators.lazy_property
    def nearest_winner_distances(self):
        """array shaped (a,)
            Preference distances of nearest winner for `a` voters. 
        """
        distances = vcalcs.voter_distances(
                                           self.voters,
                                           self.winners,
                                           self.weights
                                           )
        index = np.arange(self._num_voters)
        return distances[index, self.nearest_winners]
    
    
    @utilities.decorators.lazy_property
    def regret(self):
        """float
            Average voter regret for his nearest winner
        """
        distances = self.nearest_winner_distances
        num_voters = self._num_voters
        num_winners = self._num_winners

        regret = np.sum(distances) / num_voters
        regret = regret * num_winners
        return regret
    
    
    @utilities.decorators.lazy_property
    def nearest_regrets(self):
        """float:
            Net regrets for each winner
        """
        num_voters = self._num_voters
        num_winners = self._num_winners
        
        sregrets = []
        for ii in range(num_winners):
            index = (ii == self.nearest_winners)
            distances = self.nearest_winner_distances[index]
            regret = np.sum(distances)
            sregrets.append(regret)
        
        sregrets = np.array(sregrets) / num_voters * num_winners
        return sregrets
            
        
    @utilities.decorators.lazy_property
    def std_voters(self):
        """float:
            Standard deviation of number of nearest voters for each winner
        """
        num_voters = self._num_voters
        num_winners = self._num_winners
        
        wcounts = []
        for ii in range(num_winners):
            wcount = np.sum(ii == self.nearest_winners)
            wcounts.append(wcount)
            
        voters_per_winner = num_voters / num_winners
        std = np.std(wcounts) / voters_per_winner
        return std
    
    
    @utilities.decorators.lazy_property
    def std_regret(self):
        """float:
            Standard deviation of nearest regrets for each winner. An ideal
            proportional system ought to have low std deviation.
        """
        return np.std(self.nearest_regrets)
            

    
    
    
        


    

def candidate_regrets(voters, candidates, weights=None):
    """Calculate the voter regret for each candidate or winner.
    
    Parameters
    -----------
    voters : array (a, n)
        Voter preferences; n-dimensional voter cardinal preferences for n issues. 
    candidates : array (b, n)
        Candidate preferences for `b` candidates and `n`-dimensional issues.   
        
        
    Returns
    -------
    out : array (b,)
        Average preference distance of voters from each candidate numbering `b`.
    """
    
    voters = np.atleast_2d(voters)
    candidates = np.atleast_2d(candidates)
    num_voters = len(voters)

    # distance shape (a, b) for `a` num voters, `b` num candidates. 
    distances = vcalcs.voter_distances(voters, candidates, weights=weights)
    avg_distances = np.sum(distances, axis=0) / num_voters
        
    return avg_distances
    

def consensus_regret(voters, winners, _distances=None):
    """
    Measure overall satisfaction of all winners for all voters. 
    
    Parameters
    -----------
    voters : array, shape (a, n)
        Voter preferences; n-dimensional voter cardinal preferences for n issues. 
    winners : array, shape (b, n)
        Winner preferences for `b` winners and `n`-dimensional issues.   
        
    Returns
    -------
    regret : float
        Consensus voter regret metric
    """
    num_winners = len(winners)
    if _distances is not None:
        distances = _distances
    else:
        distances = candidate_regrets(voters, winners)
    regret = np.sum(distances) / num_winners
    return regret


def mean_regret(voters, weights=None):
    """
    Parameters
    -----------
    voters : array, shape (a, n)
        Voter preferences; n-dimensional voter cardinal preferences for n issues. 
    weights : array, shape (a, n)
        Voter preference weights for each preference. (ie, voters care
        more about some preferences than others).
        
    """
    num = len(voters)
    center = np.mean(voters, axis=0)
    
    if weights is None:
        diff = voters - center
    else:
        diff = (voters - center) * weights
    dist = np.sum(np.linalg.norm(diff, axis=1)) / num
    return dist


def median_regret(voters, weights=None):
    num = len(voters)
    center = np.median(voters, axis=0)
    if weights is None:
        diff = voters - center
    else:
        diff = (voters - center) * weights
    dist = np.sum(np.linalg.norm(diff, axis=1)) / num
    return dist


def regret_std(voters, meanvoter=None, weights=None):
    if meanvoter is None:
        v_mean = np.mean(voters, axis=0)
    else:
        v_mean = meanvoter
    v_dist = vcalcs.voter_distances(voters, 
                                    v_mean[None, :],
                                    weights=weights)
    std = np.std(v_dist)
    return std
    

#
#def _ballot_stats(self, election):
#    
#    scores = election.scores
#    ranks = election.ranks
#    ratings = election.ratings
#    
#    num_scored = np.sum(scores > 0, axis=1)
#    num_ranked = np.sum(ranks > 0, axis=1)
#    num_rated = np.sum(ratings > 0, axis=1)
#    
#    self.avg_num_rated = np.average(num_rated)
#    self.avg_num_scored = np.average(num_scored)
#    self.avg_num_ranked = np.average(num_ranked)
#    self.std_num_scored = np.std(num_scored)
#    self.std_num_ranked = np.std(num_ranked)    
#    
#    

        
class ElectionStats(object):
    """Calculate and store various regret metrics
    
    Parameters
    ----------
    voters : array, shape (a, n)
        Voter preferences; n-dimensional voter cardinal preferences for n issues. 
        
    candidates : array (b, n)
        Candidate preferences for `b` candidates and `n`-dimensional issues.      
        
    winners : array, shape (b, n)
        Winner preferences for `b` winners and `n`-dimensional issues. 
        
    ballots : array, shape (a, b)
        Submitted ballots, rows as voters & columns as candidates
        
        - Zero data ballots are minimum score, unranked, or not chosen candidates.
        
                
    Attributes
    ----------
    stats : dict
        - Election statistics
        
    stat descriptions
    ------------------
            - voter.regret.mean

        
        
    PR_regret : float
        Regret metric; See func `PR_regret`
    consensus_regret : float
        Regret metric; See func `consensus_regret`
    ideal_regret : float
        Best possible consensus_regret for this election.
    candidate_regrets : array shaped (b,)
        Preference distances from average voter for all candidates. 
        
    """
    def __init__(self, voters=None, weights=None,
                 candidates=None, winners=None, ballots=None):
        self.stats = {}
        self.run(voters=voters,
                 weights=weights, 
                 candidates=candidates, 
                 winners=winners,
                 ballots=ballots)
        return
    
        
    def run(self, voters=None, weights=None,
            candidates=None, winners=None, ballots=None):
        
        d = self.stats.copy()
        if voters is not None:
            stats1 = self._voter_stats(voters, weights)
            d.update(stats1)
            self.stats = d
        
        if (candidates is not None) and (winners is not None):
            stats2 = self._result_stats(voters, candidates, winners, weights)
            d.update(stats2)   
            
        if ballots is not None:
            stats3 = self._ballot_stats(ballots)
            d.update(stats3)
        
        self.stats = d        
        return
    
    
    def __getitem__(self, key):
        return self.stats[key]
    
    
    def keys(self):
        return self.stats.keys()

    
    def _voter_stats(self, voters, weights):
        
        regret_mean = mean_regret(voters, weights)
        regret_median = median_regret(voters, weights)    
        voter_mean = np.mean(voters, axis=0)
        voter_median = np.median(voters, axis=0)  
        voter_std = np.std(voters, axis=0)
        regret_std = self._regret_std(voters, voter_mean, weights=weights)
        
        v = {}
        v['voter.regret.mean'] =  regret_mean
        v['voter.regret.median'] =  regret_median
        v['voter.regret.std'] =  regret_std
        v['voter.mean'] =  voter_mean
        v['voter.median'] =  voter_median
        v['voter.std'] =  voter_std
        return v
    
                
    
    def _result_stats(self, voters, candidates, winners, weights):
        regret_candidates = candidate_regrets(voters, candidates)
        rr = np.mean(regret_candidates)
        
        ### Average satisfaction of voter to closest winner
        
        pr = PrRegret(voters, winners, weights)
        
        ### Overall satisfaction of all voters for all winners
        winner_pref = candidates[winners]
        winner_num = len(winners)        
        ii = winners
        rc = consensus_regret(voters,
                             winner_pref,
                             _distances=regret_candidates[ii])
        
        ### Minimum possible consensus regret for this election

        
        regret_best, candidate_best = self._regret_best(regret_candidates, winner_num)
        vse = self._vse(rc, rr, regret_best)
        
        rvm = self.stats['voter.regret.median']
        regret_median_acc = self._regret_median_accuracy(rc, rvm)
        
        d = {}
        d['regret.candidates'] = regret_candidates
        d['regret.PR'] = pr.regret
        d['regret.PR_std'] = pr.std_regret
        
        d['regret.consensus'] = rc
        d['regret.best'] = regret_best
        d['regret.random'] = rr
        d['regret.vse'] = vse
        d['regret.vsp'] =  self._satisfaction_population(rc, regret_best, rvm)
        d['regret.median_accuracy'] = regret_median_acc
        d['winners.num'] = winner_num
        d['winners'] = winners
        d['candidates.preference'] = candidates
        d['candidates.best'] = candidate_best
        return d
    
    
    def _ballot_stats(self, ballots):
        
        ballots = np.atleast_2d(ballots)
        ballot_num, candidate_num = ballots.shape
        
        # Get number of candidates marked for each ballot
        marked_array = np.sum(ballots > 0, axis=1)
        
        # Get ballots where bullet voting happened
        bullet_num = np.sum(marked_array == 1)
        bullet_ratio = bullet_num / ballot_num
        
        #Get ballots where all but one candidate is marked
        full_num = np.sum(marked_array == (candidate_num - 1))
        full_ratio = full_num / ballot_num
        
        marked_num = np.sum(marked_array)
        marked_avg = np.mean(marked_array)
        marked_std = np.std(marked_array)
        marked_ratio = marked_num / ballot_num
        
        d = {}
        d['ballot.bullet.num'] = bullet_num
        d['ballot.bullet.ratio'] = bullet_ratio
        d['ballot.full.num'] = full_num
        d['ballot.full.ratio'] = full_ratio
        d['ballot.marked.num'] = marked_num
        d['ballot.marked.avg'] = marked_avg
        d['ballot.marked.std'] = marked_std
        d['ballot.marked.ratio'] = marked_ratio
        return d
    
        
        
    
    @staticmethod
    def _regret_std(voters, meanvoter, weights=None):
        
        #v_mean = np.mean(voters, axis=0)
        v_mean = meanvoter
        v_dist = vcalcs.voter_distances(voters,
                                        v_mean[None, :],
                                        weights=weights)
        std = np.std(v_dist)
        return std
    
    
    @staticmethod
    def _regret_best(candidate_regrets, winner_num):
        """Retrieve best regrests and corresponding winner indices"""
        ii = np.argsort(candidate_regrets)
        ii_ideal = ii[0 : winner_num] 
        ri = np.mean(candidate_regrets[ii_ideal])    
        return ri, ii_ideal
    
    
    @staticmethod
    def _vse(regret, regret_random, regret_best):
        """Calculate voter satisfaction efficiency
        
        Parameters
        -----------
        regret : float
            Average voter regret for winners; consensus regret
        regret_random : float
            Average voter regret for all candidates
        regret_best : float
            Best candidate regret of election
        """
        
        U = regret
        R = regret_random
        O = regret_best
        vse = (U - R) / (O - R)
        return vse
    
    @staticmethod
    def _satisfaction_population(regret, regret_best, regret_median):
        """Voter satisfaction normalizd by ideal centroid population regret"""
        U = regret
        O = regret_best
        M =  regret_median
        
        vp = 1 - (U - O) / M
        return vp
    

    
    
    @staticmethod
    def _regret_median_accuracy(regret, regret_median):
        """
        Calculate voter method accuracy tending towards the population median
        
        Parameters
        -----------
        regret : float
            Average voter regret for winners; consensus regret
        regret_median : float
            Voter regret if median voter elected as winner. 
        """
        U = regret
        M = regret_median
        return 1.0 - abs(U - M) / M
        

    
    


if __name__ == '__main__':
    rs = np.random.RandomState(None)
    win_num = np.arange(5, 100, 1)
    regrets = []
    ndim = 1
    for w in win_num:
        voters = rs.rand(5000, ndim) * 10 
        winners = rs.rand(w, ndim) * 10 
        r = PR_regret(voters, winners)
        regrets.append(r)
    
    
    import matplotlib.pyplot as plt
    plt.plot(win_num, regrets)
    plt.ylim(0, None)
    
    
    
    

