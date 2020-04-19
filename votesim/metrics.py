# -*- coding: utf-8 -*-
"""
Various metrics to measure performance of an election
"""
import logging


import numpy as np
import scipy
import scipy.interpolate
from scipy.interpolate import NearestNDInterpolator

from votesim import utilities
from votesim.models import vcalcs
from votesim.utilities.math import NearestManhattanInterpolator
logger = logging.getLogger(__name__)





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
    norm : int 
        - 1 for Manhattan traffic-style norm
        - 2 for Euclidean distance norm
    
    """
    
    def __init__(self, voters, winners, weights=None, norm=1):
        self.voters = np.atleast_2d(voters)
        self.winners = np.atleast_2d(winners)       
        self.weights = weights
        self._norm = norm
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
            if self._norm == 1:
                interp  = NearestManhattanInterpolator(winners, winner_ids)
            elif self._norm == 2:
                interp = NearestNDInterpolator(winners, winner_ids)
            else:
                raise ValueError('Only norms 1 and 2 are supported')
            
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
    

def candidate_regrets(voters, candidates, weights=None, order=1):
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
    distances = vcalcs.voter_distances(voters,
                                       candidates, 
                                       weights=weights,
                                       order=order)
    
    avg_distances = np.sum(distances, axis=0) / num_voters
    return avg_distances



def voter_regrets(voters, weights=None, order=1, pnum=10, maxsize=5000, seed=None):
    """Calculate the voter regrets for each other if voters became a candidate
    
    Parameters
    ----------
    voters : array shape (a, n)
        Voter preferences; `a` number of voters, cardinal preferences for `n` issues.     
    weights : None or array shape (a, n)
        Dimensional weightings of each voter for each dimension.
        Only relevant if n > 1    
    order : int
        Order of norm
        
        * 1 = taxi-cab norm; preferences for each issue add up
        * 2 = euclidean norm; take the sqrt of squares. 
    pnum : int
        Number of voters to calculate distances for at-a-time, for memory issues
    maxsize: int
        For large populations this calculation is expensive. Use this to sample
        a subset of the voter population. Default 5000.        
        Set to None to use all voters. 
        
    Returns
    -------
    out : array shape (c,)
        Voter regrets for each voter as a proposed candidate. 
        
        - c = a if maxsize <= number voters or maxsize==None
        - c = maxsize otherwise for sampled voters.
    
    """
    
    cnum = len(voters)
    if maxsize is not None:
        if cnum > maxsize:
            rs = np.random.RandomState(seed)
            ii = rs.choice(cnum, size=maxsize, replace=False)
            voters = voters[ii]
    
    numbers = np.arange(0, cnum + pnum, pnum)
    lb_nums = numbers[0:-1]
    ub_nums = numbers[1:]

    
    dlist = []
    for lb, ub in zip(lb_nums, ub_nums):
        candidatesi = voters[lb : ub]
        try:
            d = candidate_regrets(voters, candidatesi, weights=weights, order=order)
            dlist.append(d)
        except MemoryError:
            return voter_regrets(voters, weights, order, pnum=1)
        
    return np.concatenate(dlist)
    
    

def consensus_regret(voters, winners, _distances=None):
    """
    Measure overall average satisfaction of all winners for all voters. 
    
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


def mean_regret(voters, weights=None, order=1):
    """
    Measure overall regret of voters if a candidate located at the centroid 
    was elected.
    
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
    dist = np.sum(np.linalg.norm(diff, axis=1, ord=order)) / num
    return dist


def median_regret(voters, weights=None, order=1):
    num = len(voters)
    center = np.median(voters, axis=0)
    if weights is None:
        diff = voters - center
    else:
        diff = (voters - center) * weights
    dist = np.sum(np.linalg.norm(diff, axis=1, ord=order)) / num
    return dist


def regret_std(voters, meanvoter=None, weights=None, order=1):
    if meanvoter is None:
        v_mean = np.mean(voters, axis=0)
    else:
        v_mean = meanvoter
    v_dist = vcalcs.voter_distances(voters, 
                                    v_mean[None, :],
                                    weights=weights,
                                    order=order)
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
    order : int
        Order of regret distance calculations (default 1)
        
    Usage
    ------
    To access all metrics, use
    
    >>> self.get_dict()
    
    
    To retrieve descriptions for all matrics, use
    
    >>> self.get_docs()
    
        
    """
    def __init__(self, voters=None, weights=None,
                 candidates=None, winners=None, ballots=None, order=1):
        self.stats = {}
        self._voters = voters
        self._weights = weights
        self._candidates = candidates
        self._winners = winners
        self._ballots = ballots
        self._order = order
        
        self._cache_voter = {}
        self._cache_candidate = {}
        self._cache_result = {}
        
#        self.run(voters=voters,
#                 weights=weights, 
#                 candidates=candidates, 
#                 winners=winners,
#                 ballots=ballots)
        return
    
    
    def set(self,
            voters=None,
            weights=-1,
            candidates=None,
            winners=None,
            ballots=None):
        """Set voters, weights, candidates or winners to recalculate"""
        
        if voters is not None:
            self._voters = voters
            self._cache_voters = {}
            self._cache_candidate = {}
            self._cache_result = {}
            
            
        if weights != -1:
            self._weights = weights
            self._cache_voters = {}
            self._cache_candidate = {}
            self._cache_result = {}            
            
        if candidates is not None:
            self._candidates = candidates
            self._cache_candidate = {}
            self._cache_result = {}
        
        if winners is not None:
            self._winners = winners
            self._cache_result = {}
            
        if ballots is not None:
            self._ballots = ballots
            
            
            
            
        
#    def run(self, voters=None, weights=None,
#            candidates=None, winners=None, ballots=None):
#        
#        d = self.stats.copy()
#        if voters is not None:
#            stats1 = self._voter_stats(voters, weights)
#            d.update(stats1)
#            self.stats = d
#        
#        if (candidates is not None) and (winners is not None):
#            stats2 = self._result_stats(voters, candidates, winners, weights)
#            d.update(stats2)   
#            
#        if ballots is not None:
#            stats3 = self._ballot_stats(ballots)
#            d.update(stats3)
#        
#        self.stats = d        
#        return
            
            
            
    def _get_category_keys(self, category):
        """Divide metrics into categories defined by an attribute prefix"""
        a = dir(self)
        prefix = category + '_'
        
        new = {}
        for name in a:
            if name.startswith(prefix):
                newkey = name.replace(prefix, category + '.')
                new[newkey] = name
        return new
    
    
    @property        
    def _keys_voter(self):
        """Retrieve voter metrics' attribute names"""
        category = 'voter'
        return self._get_category_keys(category)
        
    @property        
    def _keys_candidate(self):
        """Retrieve candidate metrics' attribute names"""
        category = 'candidate'
        return self._get_category_keys(category)
    
    @property        
    def _keys_regret(self):
        """Retrieve regret metrics' attribute names"""
        category = 'regret'
        return self._get_category_keys(category)
        
    @property        
    def _keys_winner(self):
        """Retrieve winner metrics' attribute names"""
        category = 'winner'
        return self._get_category_keys(category)

    @property        
    def _keys_ballot(self):
        """Retrieve ballot metrics' attribute names"""
        category = 'ballot'
        return self._get_category_keys(category)

            
    @utilities.lazy_property
    def _keydict(self):
        """Retrieve dict keynames that retrieve attribute data"""
        new = {}
        new.update(self._keys_voter)
        new.update(self._keys_candidate)
        new.update(self._keys_regret)
        new.update(self._keys_winner)
        new.update(self._keys_ballot)
        return new
    
    
    def get_keys(self):
        """Get a list of all available statistics"""
        return list(self._keydict.keys())
                
                
    def get_dict(self):
        """Retrieve all available statistics"""
        new = {}
        for key, attrname in self._keydict.items():
            try:
                new[key] = getattr(self, attrname)
            except RuntimeError:
                pass
        return new
            
    
    def get_docs(self):
        """Retrieve all available statistic descriptions as dict"""
        clss = type(self)
        new = {}
        for key, attrname in self._keydict.items():
            doc = getattr(clss, attrname).__doc__
            doc = doc.replace('\n', ' ')
            doc = ' '.join(doc.split())
            new[key] =  doc
        return new

               
    @property
    def voters(self):
        if self._voters is None:
            raise RuntimeError('Voters are not yet defined in Metrics')
        return self._voters
    
    @property
    def weights(self):
        return self._weights
    
    
    @property
    def candidates(self):
        if self._candidates is None:
            raise RuntimeError('Candidates are not yet defined in Metrics')        
        return self._candidates
    
    
    @property
    def winners(self):
        if self._winners is None:
            raise RuntimeError('Winners are not yet defined in Metrics')              
        return self._winners
    
    
    @property
    def ballots(self):
        if self._ballots is None:
            raise RuntimeError('Ballots are not yet defined in Metrics')          
        return self._ballots
    
    
    ### Metrics
    
    
    @utilities.lazy_property2('_cache_voter')
    def voter_regret_mean(self):
        """Regret of voters if winner is located at preference mean"""
        return mean_regret(self.voters, self.weights, order=self._order)
    
    
    @utilities.lazy_property2('_cache_voter')
    def voter_regret_median(self):
        """Regret of voters if winner is located at preference median"""
        return median_regret(self.voters, self.weights)
    
    
    @utilities.lazy_property2('_cache_voter')
    def voter_regret_random_avg(self):
        """Average regret of voters if winner is randomly selected from voter 
        population"""
        
        r = voter_regrets(self.voters,
                          self.weights,
                          maxsize=5000,
                          order=self._order,
                          seed=0)
        return np.mean(r)
    
    
    @utilities.lazy_property2('_cache_voter')
    def voter_mean(self):
        """"array shape (n) : Preference mean of voters for n preference dimensions"""
        return np.mean(self.voters, axis=0)
    
    
    @utilities.lazy_property2('_cache_voter')
    def voter_median(self):
        """array shape (n) : Preference median of voters for n preference dimensions"""
        return np.median(self.voters, axis=0)
    
    
    @utilities.lazy_property2('_cache_voter')
    def voter_std(self):
        """array shape (n) : Preference standard deviation of voters for
        n preference dimensions"""
        return np.std(self.voters, axis=0)    
    
    
    @utilities.lazy_property2('_cache_voter')
    def voter_regret_std(self):
        """Standard deviation of regret """
        meanvoter = self.voter_mean
        return regret_std(self.voters,
                          meanvoter=meanvoter,
                          weights=self.weights,
                          order=self._order)
    
    
    @utilities.lazy_property2('_cache_candidate')
    def candidate_regrets(self):
        """array shape (c) : voter regret for each candidate"""
        return candidate_regrets(self.voters, 
                                 self.candidates,
                                 order=self._order)
    
    
    @utilities.lazy_property2('_cache_result')
    def _PR_regret(self):
        pr = PrRegret(self.voters, self.winners, self.weights)
        regret = pr.regret
        std_regret = pr.std_regret        
        return regret, std_regret
    
    
    @property
    def regret_PR(self):
        """Multi-winner average regret for Proportional Representation.
        Average voter regret for his nearest winner"""
        return self._PR_regret[0]
    
    
    @property
    def regret_PR_std(self):
        """Standard deviation of nearest regrets for each winner. An ideal
        proportional system ought to have low std deviation"""
        return self._PR_regret[1]
    
    
    @utilities.lazy_property2('_cache_result')
    def winner_num(self):
        """Number of winners for this election"""
        return len(self.winners)
    
    
    @property
    def winner_all(self):
        """All winners of election"""
        return self.winners


    @utilities.lazy_property2('_cache_result')
    def regret_consensus(self):
        """overall satisfaction of all winners for all voters."""
        candidate_regrets = self.candidate_regrets
        ii = self.winners
        
        winner_pref = self.candidates[ii]
        
        rc = consensus_regret(self.voters,
                             winner_pref,
                             _distances=candidate_regrets[ii]) 
        return rc


    @utilities.lazy_property2('_cache_candidate')
    def _regret_best(self):
        """Retrieve best regrests and corresponding winner indices"""
        candidate_regrets = self.candidate_regrets
        winner_num = self.winner_num
        
        ii = np.argsort(candidate_regrets)
        ii_ideal = ii[0 : winner_num] 
        ri = np.mean(candidate_regrets[ii_ideal])    
        return ri, ii_ideal
    
    
    @property
    def regret_best(self):
        """Best possible regret for the best candidate in election"""
        return self._regret_best[0]
    

    @property
    def candidate_best(self):
        """Best possible candidate (in terms of regret) in election"""
        return self._regret_best[1]
    
    
    @utilities.lazy_property2('_cache_candidate')
    def candidate_regret_random(self):
        """Average regret if a random candidate became winner"""
        return np.mean(self.candidate_regrets)
    
        
    @property
    def candidate_preference(self):
        """Preference locations of candidates"""
        return self.candidates
    
    
    @property
    def regret_efficiency_candidate(self):
        """Voter satisfaction efficiency, compared to random candidate"""    
        U = self.regret_consensus
        R = self.candidate_regret_random
        B = self.regret_best
        vse = (U - R) / (B - R)
        return vse        
    
    
    @property
    def regret_efficiency_voter(self):
        """My updated satisfaction efficiency equation normalizing to voter population
        rather than candidate population"""
        U = self.regret_consensus
        R2 = self.voter_regret_random_avg
        R1 = self.voter_regret_median
        B = self.regret_best

        return 1.0 - abs(U - B) / (R2 - R1)
    
    
    @property
    def regret_normed(self):
        """Voter regret normalized to ideal"""
        U = self.regret_consensus
        R = self.voter_regret_median
        return U / R - 1
    
    
    
    
    
    @utilities.lazy_property2('_cache_ballot')
    def _ballot_stats(self):
        ballots = np.atleast_2d(self.ballots)
        ballot_num, candidate_num = ballots.shape
        
        # Get number of candidates marked for each ballot
        marked_array = np.sum(ballots > 0, axis=1)
        
        # Get ballots where bullet voting happened
        bullet_num = np.sum(marked_array == 1)
        bullet_ratio = bullet_num / ballot_num
        
        
        #Get ballots where all but one candidate is marked
        full_num = np.sum(marked_array >= (candidate_num - 1))
        full_ratio = full_num / ballot_num
        
        marked_num = np.sum(marked_array)
        marked_avg = np.mean(marked_array)
        marked_std = np.std(marked_array)
        
        
        d = {}
        d['ballot.bullet.num'] = bullet_num
        d['ballot.bullet.ratio'] = bullet_ratio
        d['ballot.full.num'] = full_num
        d['ballot.full.ratio'] = full_ratio
        d['ballot.marked.num'] = marked_num
        d['ballot.marked.avg'] = marked_avg
        d['ballot.marked.std'] = marked_std
        return d
    
    
    @property
    def ballot_bullet_num(self):
        """Number of ballots where voters only bullet voted for 1 candidate"""
        return self._ballot_stats['ballot.bullet.num']

    
    @property
    def ballot_bullet_ratio(self):
        """Ratio of ballots where voters only bullet voted for 1 candidate"""
        return self._ballot_stats['ballot.bullet.ratio']
    
    
    @property
    def ballot_full_num(self):
        """Number of ballots where all but one candidate is marked"""
        return self._ballot_stats['ballot.bullet.ratio']
    
        
    @property
    def ballot_full_ratio(self):
        """Ratio of ballots where all but one candidate is marked"""
        return self._ballot_stats['ballot.bullet.ratio']
    
        
    @property
    def ballot_marked_num(self):
        """Total number of marked candidates for all ballots"""
        return self._ballot_stats['ballot.marked.num']
    
    
    @property
    def ballot_marked_avg(self):
        """Average number of marked candidates per ballot"""
        return self._ballot_stats['ballot.marked.avg']
    
        
    @property
    def ballot_marked_std(self):
        """Std deviation of marked candidates per ballot"""
        return self._ballot_stats['ballot.marked.std']

    
#    def _result_stats(self, voters, candidates, winners, weights):
#        regret_candidates = candidate_regrets(voters, candidates)
#        rr = np.mean(regret_candidates)
#        
#        ### Average satisfaction of voter to closest winner
#        
#        pr = PrRegret(voters, winners, weights)
#        
#        ### Overall satisfaction of all voters for all winners
#        winner_pref = candidates[winners]
#        winner_num = len(winners)        
#        ii = winners
#        rc = consensus_regret(voters,
#                             winner_pref,
#                             _distances=regret_candidates[ii])
#        
#        ### Minimum possible consensus regret for this election
#
#        
#        regret_best, candidate_best = self._regret_best(regret_candidates, winner_num)
#        vse = self._vse(rc, rr, regret_best)
#        
#        rvm = self.stats['voter.regret.median']
#        regret_median_acc = self._regret_median_accuracy(rc, rvm)
#        
#        d = {}
#        d['regret.candidates'] = regret_candidates
#        d['regret.PR'] = pr.regret
#        d['regret.PR_std'] = pr.std_regret
#        
#        d['regret.consensus'] = rc
#        d['regret.best'] = regret_best
#        d['regret.random'] = rr
#        d['regret.vse'] = vse
#        d['regret.vsp'] =  self._satisfaction_population(rc, regret_best, rvm)
#        d['regret.median_accuracy'] = regret_median_acc
#        d['winners.num'] = winner_num
#        d['winners'] = winners
#        d['candidates.preference'] = candidates
#        d['candidates.best'] = candidate_best
#        return d
#    

    

#
#if __name__ == '__main__':
#    rs = np.random.RandomState(None)
#    win_num = np.arange(5, 100, 1)
#    regrets = []
#    ndim = 1
#    for w in win_num:
#        voters = rs.rand(5000, ndim) * 10 
#        winners = rs.rand(w, ndim) * 10 
#        r = PR_regret(voters, winners)
#        regrets.append(r)
#    
#    
#    import matplotlib.pyplot as plt
#    plt.plot(win_num, regrets)
#    plt.ylim(0, None)
#    
#    
    
    

