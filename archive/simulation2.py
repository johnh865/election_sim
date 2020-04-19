# -*- coding: utf-8 -*-
"""
Simulate elections. 


Elements of an election

1. Create voter preferences

 - Create voter preference distributions
 - Create voter preference tolerance distribution

2. Create candidate preferences
3. Simulate voter behavior, strategy
4. Transform voter preferences into candidate scores or rankings
5. Input scores/ranks into election system.
6. Run the election.
7. Measure the results. 


"""
import itertools

from votesim import metrics, randomstate, behavior
from votesim.votesystems import irv, plurality, score, condorcet
from votesim.utilities import utilities

#
#from . import metrics, randomstate, behavior
#from .votesystems import irv, plurality, score, condorcet
#from .utilities import utilities

import scipy
from scipy.interpolate import NearestNDInterpolator
from scipy.stats import truncnorm


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import seaborn as sns

import logging
logger = logging.getLogger(__name__)

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
    if rstate is None:
        rstate = randomstate.state
    
    winners = np.array(winners)
    num_found = len(winners)
    num_needed =  numwinners - num_found
    
    if num_needed > 0:
        
        new = rstate.choice(ties, size=num_needed, replace=False)
        winners = np.append(winners, new)
    return winners.astype(int)



#
#
#def output_stats(output, bins=10):
#    winners = output['winners']
#    pref_candidates = output['candidate preferences']
#    pref_voters = output['voter preferences']
#    pref_winners = pref_candidates[winners]
#    num_voters = len(pref_voters)
#    num_candidates = len(pref_candidates)
#    num_winners = len(winners)
#    
#    
#    h_voters, h_edges = hist_norm(pref_voters, bins=bins)
#    h_edges_c = np.copy(h_edges)
#    h_edges_c[0] = pref_candidates.min()
#    h_edges_c[-1] = pref_candidates.max()
#    
#    
#    h_candidates, _ = hist_norm(pref_candidates, h_edges_c)
#    h_winners, _ = hist_norm(pref_winners, h_edges)
#    
#    hist_error = np.sum(np.abs(h_winners - h_voters))
#    avg_voter = np.mean(pref_voters)
#    avg_winner = np.mean(pref_winners)
#    avg_error = avg_voter - avg_winner
#    
#    std_voter = np.std(pref_voters)
#    std_winner = np.std(pref_winners)
#    std_error = std_voter - std_winner
#    
#    median_voter = np.median(pref_voters)
#    median_winner = np.median(pref_winners)
#    median_error = median_voter - median_winner
#    
#    
#
#    
#    return locals()


#class ElectionStats(object):
#    """Store election statistics in this sub-class of Election
#    
#    Attributes
#    ---------
#    num_voters : int
#        Number of voters participating in election
#    num_candidates: int
#        Number of candidates participating in election
#    num_winners : int
#        Number of winners for this election
#        
#    avg_voter : float
#        Average preference of all voters
#    avg_winner : float
#        Average preference of all winning candidates
#    avg_error : float
#        Preference difference between average voter and average winner. 
#        
#    std_voter : float
#        Preference standard deviation for all voters
#    """
#    def __init__(self, pref_winners, pref_candidates, pref_voters):
#        
#        self.num_voters = len(pref_voters)
#        self.num_candidates = len(pref_candidates)
#        self.num_winners = len(pref_winners)        
#        
#        self.avg_voter = np.mean(pref_voters, axis=0)
#        self.avg_winner = np.mean(pref_winners, axis=0)
#        self.avg_error = self.avg_voter - self.avg_winner
#            
#        self.std_voter = np.std(pref_voters)
#        self.std_winner = np.std(pref_winners)
#        self.std_error = self.std_voter - self.std_winner
#        
#        self.median_voter = np.median(pref_voters)
#        self.median_winner = np.median(pref_winners)
#        self.median_error = self.median_voter - self.median_winner        
#            
#        regret, voter_std = PR_regret(pref_voters, pref_winners)
#        self.regret = regret
#        self.regret_std_num = voter_std        
#        
#        return
#
#    
#        



def ltruncnorm(loc, scale, size, random_state=None):
    """
    Truncated normal random numbers, cut off at locations less than 0.
    
    Parameters
    -----------
    loc : float 
        Center coordinate of gaussian distribution
    scale : float 
        Std deviation scale
    size : int
        Number of random numbers to generate
    random_state : None or numpy.random.RandomState 
        Random number seeding object, or None.
        
    Returns
    ---------
    out : array shaped (size)
        Output samples
    """
    xmin = -loc / scale
    t = truncnorm(xmin, 1e6)
    s = t.rvs(size=size, random_state=random_state)
    s = s * scale  + loc
    return s




def gaussian_preferences(coords, sizes, scales, rstate=None):
    """
    Generate gaussian preference distributions at coordinate and specified size
    
    
    Parameters
    ----------
    coords : array shaped (a, b) 
        Centroids of a faction voter preferences.
    
        - rows `a` = coordinate for each faction
        - columns `b' = preference dimensions. The more columns, the more preference dimensions. 
        
    sizes : array shaped (a,)
        Number of voters within each faction, with a total of `a` factions.
        Use this array to specify how many people are in each faction.
    scales : array shaped (a, b)
        The preference spread, width, or scale of the faction. These spreads
        may be multidimensional. Use columns to specify additional dimensions. 
        
        
    Returns
    -------
    out : array shaped (c, b)
        Population preferences of `c` number of voters in `b` preference dimensions.
        

    """
    if rstate is None:
        rstate = randomstate.state
    new = []
    coords = np.atleast_2d(coords)
    ndim = coords.shape[1]
    for center, size, scale in zip(coords, sizes, scales):
        logger.debug('size=%s', size)
        pi = rstate.normal(loc=center,
                           scale=scale,
                           size=(size, ndim))
        new.append(pi)
    new = np.vstack(new)
    return new        
        
        
def generate_preferences(numfactions, size, ndim=1, sepfactor=1, rstate=None):
    """
    Create multi-peaked gaussian distributions of preferences
    
    Parameters
    ----------
    numvoters : int array of shape (a,), or int
        Number of voter preferences to generate. If list/array, generate 
        multiple gaussian voter preference peaks. 
    ndim : int, default=1
        Number of preference dimensions to consider
    sepfactor : float
        Scaling factor of typical distance of peak centers away from one another
    seed : None (default) or int
        Convert preference generation to become deterministic & pseudo-random
        for testing purposes. 
        
        - If None, use internal clock for seed generation
        - If int, use this seed to generate future random variables. 
        
    Returns
    -------
    out : array shaped (c, ndim)
        Voter preferences for ndim number of dimensions.
        
    Example
    -------
    
    Create a 2-dimensional preference spectrum, with 3 factions/preference 
    peaks with:
        - 200 voters in the 1st faction
        - 400 voters in the 2nd faction
        - 100 voters in the 3rd faction
    
    >>> p = generate_voter_preference((200, 400, 100), 2)
    
    Create a 1-dimensional preference spectrum with gaussian distribution
    for 500 voters.
    
    >>> p = generate_voter_preferences(500, 1)
    
    """
    if rstate is None:
        rstate = randomstate.state
    
    size1 = int(size/3)
    numvoters = [rstate.randint(size1, size) for i in range(numfactions)]
    new = []
    numvoters = np.atleast_1d(numvoters)
#    numleft = numvoters
    for pop_subset in numvoters:
        
#        if i == numpeaks - 1:
#            pop_subset = numleft
#        else:
#            pop_subset = np.random.randint(0, numleft)
        
        center = (rstate.rand(ndim) - 0.5) * sepfactor
        scale = rstate.rayleigh(size=ndim) / 2
        pi = rstate.normal(loc=center,
                              scale=scale, 
                              size=(pop_subset, ndim))
        new.append(pi)
#        numleft = numleft - pop_subset
    new = np.vstack(new)
    return new
        



class Voters(object):
    
    
    
class ElectionRun(object):
    """
    Parameters
    ----------
    voters : array shape (a, n)
        Voter preferences; n-dimensional voter cardinal preferences for n issues. 
    candidates : array shape (b, n)
        Candidate preferences for b-candidates and n-dimensional issues. 
    tol : float, or array shaped (a,)
        Voter candidate tolerance distance. If cardinal preference exceed tol, 
        utility set to zero. 
    numwinners : int
        Number of winners for this election. 

    Attributes
    ----------
    ranks : array shaped (a, b)
        Voter rankings for each candidate
    ratings : array shaped (a, b)
        Voting ratings for each candidate
    scores : array shaped (a, b)
        Voter integer scores from 1 to `scoremax` for each candidate
    votes : array shaped (a, b)
        FPTP, plurality ballots for each candidate
    stats : `ElectionStats` object
        Statistics of voters and candidates
        
    """    
    
    def __init__(self, voters, candidates,
                 numwinners=1, cnum=None, tol=0.5, 
                 error=0, scoremax=5, strategy='abs',
                 kwargs=None):
        
        self.voters = np.array(voters)
        self.candidates = np.array(candidates)
        self.numwinners = numwinners
        
        self.cnum = cnum
        self.scoremax = scoremax
        self.tol = tol
        self.error=error
        self.strategy = strategy        
        
        self._run_behavior()
        return

        
    def _run_behavior(self):
        """
        Define voter behavior here. Convert voter & candidate preferences
        into candidate ratings. 
        Convert ratings into scores, ranks, and plurality votes. 
        
        """
        
        scoremax = self.scoremax
        strategy = self.strategy
        ratings = behavior.voter_scores_by_tolerance(self.voters, 
                                                    self.candidates,
                                                    tol=self.tol,
                                                    error_std=self.error,
                                                    cnum = self.cnum,
                                                    strategy=strategy)
        ranks = behavior.score2rank(ratings)
        scores = np.round(ratings * scoremax)
        self.ranks = ranks
        self.scores = scores
        self.ratings = ratings
        self.votes = behavior.getplurality(ranks=ranks)
        return
    
    
    def run(self, etype=None, method=None,
            btype=None, scoremax=None, seed=None, kwargs=None):
        """
        Run the election & obtain results. For ties, randomly choose winner. 
        
        Parameters
        ----------
        etype : str
            Name of election type.
            Mutually exclusive with `method` and `btype`
            Supports the following election types:
                
            - 'rrv' -- Reweighted range and pure score voting
            - 'irv' -- Single tranferable vote, instant runoff.
            - 'plurality' -- Traditional plurality & Single No Transferable Vote.
            
                
        method : func
            Voting method function. Takes in argument `data` array shaped (a, b)
            for (voters, candidates) as well as additional kwargs.
            Mutually exclusive with `etype`.
            
            >>> out = method(data, numwin=self.numwinneres, **kwargs)
            
        btype : str
            Voting method's ballot type. 
            Mutually exclusive with `etype`, use with `method`.
            
            - 'rank' -- Use candidate ranking from 1 (first place) to n (last plast), with 0 for unranked.
            - 'score' -- Use candidate rating/scored method.
            - 'vote' -- Use traditional, single-selection vote. Vote for one (1), everyone else zero (0).        
        
        Returns
        --------
        out : `ElectionResults` object
            Results of election, See `simulation.ElectionResults`
        """

        if kwargs is None:
            kwargs = {}    
        if scoremax is None:
            scoremax = self.scoremax
            
        ## Run canned election systems with prefilled parameters        
        if method is None:
            if etype == 'rrv':
                return self.run(etype=etype,
                                method=score.reweighted_range,
                                btype='score', 
                                kwargs=kwargs)
            elif etype == 'irv':
                return self.run(etype=etype,
                                method=irv.irv, 
                                btype='rank',
                                kwargs=kwargs)
            elif etype == 'plurality':
                return self.run(etype=etype,
                                method=plurality.plurality, 
                                btype='vote',
                                kwargs=kwargs)        
            elif etype == 'star':
                return self.run(etype=etype,
                                method=score.star, 
                                btype='score',
                                kwargs=kwargs)
            elif etype == 'smith_minimax':
                return self.run(etype=etype,
                                method=condorcet.smith_minimax, 
                                btype='rank',
                                kwargs=kwargs)
            elif etype == 'ttr':
                return self.run(etype=etype,
                                method=irv.top2runoff,
                                btype='rank',
                                kwargs=kwargs)
            elif etype == 'approval':
                return self.run(etype=etype,
                                method=score.approval50, 
                                btype='score',
                                kwargs=kwargs)            
            elif etype != None:
                raise ValueError('etype=%s not a valid election type' % etype)
            
            
        ## Run custom election systems            
        if btype == 'rank':
            self.ballots = self.ranks.copy()
            
        elif btype == 'score':
            self.ballots = np.round(self.ratings * scoremax)
            self.scores = self.ballots.copy()
            
        elif btype == 'vote':
            self.ballots = self.votes.copy()
            
        else:
            raise ValueError('btype %s not available.' % btype)      
            
        fargs = method.__code__.co_varnames
        if 'seed' in fargs:
            kwargs['seed'] = seed

        try:     
            out1 = method(self.ballots, 
                      numwin=self.numwinners, 
                      **kwargs)            
        except TypeError:
            out1 = method(self.ballots, 
                      **kwargs)           
            
        winners = out1[0]
        ties = out1[1]
        output = out1[2:]
        
        winners = handle_ties(winners, ties, self.numwinners)         
        
#        run_info = {}
#        run_info['etype'] = etype
#        run_info['method'] = method.__name__
#        run_info['btype'] = btype
#        run_info['scoremax'] = scoremax
#        run_info['kwargs'] = kwargs
#        run_info['ties'] = ties
#        run_info['winners'] = winners
#        self.run_info = run_info
        self.winners = winners
        self.ties = ties
        self.method_output = output
        
        return 
    
#    
#    def _electionresults(self, winners, ties, ):
#        self.winners = winners
#        self.ties = ties
#        if len(self.ties) > 1:
#            self.ties_found = True
#        else:
#            self.ties_found = False
#        
#        self.output = output
#        self.methodname = method.__name__
#    

#        sns.lmplot(c, yc,)
#        sns.lmplot(w, yw,)    
    
#    def run_stats(self):
#        
#        winners = self.winners
#        pref_candidates = self.candidates
#        pref_voters = self.voters
#        pref_winners = pref_candidates[winners]
#        num_voters = len(pref_voters)
#        num_candidates = len(pref_candidates)
#        num_winners = len(winners)
#        
##        pref_min = np.minimum(pref_)
#        
#        h_voters, h_edges = hist_norm(pref_voters, bins=self.bins)
#        h_edges_c = [(a[0:-1] + a[1:])/2. for a in h_edges]
#        
#        h_candidates, _ = hist_norm(pref_candidates, h_edges)
#        h_winners, _ = hist_norm(pref_winners, h_edges)
#        
#        hist_error = np.sum(np.abs(h_winners - h_voters)) / h_voters.shape
#        
#        avg_voter = np.mean(pref_voters, axis=0)
#        avg_winner = np.mean(pref_winners, axis=0)
#        avg_error = np.linalg.norm(avg_voter - avg_winner)
#        
#        std_voter = np.std(pref_voters, axis=0)
#        std_winner = np.std(pref_winners, axis=0)
#        std_error = np.linalg.norm(std_voter - std_winner)
#        
#        median_voter = np.median(pref_voters, axis=0)
#        median_winner = np.median(pref_winners, axis=0)
#        median_error = np.linalg.norm(median_voter - median_winner)
#        
#        regret, voter_std = PR_regret(pref_voters, pref_winners)
#        
#        self.stat_output = locals()
#        return self.stat_output





class Election(object):
    """
    Simulate elections. 
    
    1. Create voters
    -----------------
    Voters can be created with methods:
        
    - set_random_voters
    - set_voters
    
    2. Create candidates
    --------------------
    Candidates can be randomly created or prescribed using methods:
    
    - generate_candidates
    - add_candidates
    - add_median_candidate
    - add_faction_candidate
    
    3. Run the election
    ---------------------
    Use `run` to run an election method of your choice. All historical output
    is recorded in `self._result_history`.
    
    
    4. Retrieve election output
    ----------------------------
    Output is stored in the attributes:
        
    - self.result -- dict of dict of various metrics and arguments. 
    - self.dataseries() -- construct a Pandas Series from self.result
    - self.results() -- Pandas DataFrame of current and previously run results.  
    
    """
    def __init__(self, seeds=(None,None,None)):
        
        self._args = {}
        self.set_seed(*seeds)
        self._stats = metrics.ElectionStats()
        self._result_history = []
        
        return
    
    
    def set_user_args(self, kwargs):
        """Add user defined dict of arguments to database"""
        self._args = kwargs
    
    
    def set_seed(self, voters, candidates=None, election=None):
        """
        Set random seed for voter generation, candidate generation, and running elections. 
        """
        if candidates is None:
            candidates = voters
        if election is None:
            election = voters
            
        self._seed_voters = voters
        self._seed_candidates = candidates
        self._seed_election = election
        return
    
    

    @staticmethod
    def _RandomState(seed, level):
        """
        Create random state.
        Generate multiple random statse from a single seed, by specifying
        different levels for different parts of Election. 
        
        Parameters
        ----------
        seed : int
            Integer seed
        level : int
            Anoter integer seed.
        """
        if seed is None:
            return np.random.RandomState()
        else:
            return np.random.RandomState((seed, level))
        
    
    
    
    def set_random_voters(self, ndim, nfactions,
                        size_mean=100, 
                        size_std=1.0, 
                        width_mean=1.0,
                        width_std=0.5,
                        tol_mean = 1.0,
                        tol_std = 0.5,
                        error_std = 0.0,):
        """
        Parameters
        -----------
        ndim : int
            Number of preference dimensions
        nfactions : int
            Number of voter factions
        size_mean : int
            Average number of voters per faction
        size_std : float
            Std deviation of number of voters per faction. 
        width_mean : float
            Average preference width/scale of faction normal distribution.
        width_std : float
            Std deivation of preference width/scale of faction normal distribution.
        seed : None or int
            Random state seed
        """
        seed = self._seed_voters
        rs = self._RandomState(seed, 1)

        # generation faction centroid coordinates
        coords = rs.uniform(-1, 1, size=(nfactions, ndim))
        sizes = ltruncnorm(
                           loc=size_mean,
                           scale=size_std * size_mean, 
                           size=nfactions,
                           random_state=rs
                           )
        sizes = sizes.astype(int)
        sizes = np.maximum(1, sizes)  # Make sure at least one voter in each faction
        
        widths = ltruncnorm(
                            loc=width_mean,
                            scale=width_std,
                            size=nfactions * ndim
                            )
        
        widths = np.reshape(widths, (nfactions, ndim))
        
        logger.debug('coords=\n %s', coords)
        logger.debug('sizes=\n %s', sizes)
        logger.debug('widths=\n %s', widths)

        self.set_voters(coords, 
                        sizes, 
                        widths,
                        tol_mean=tol_mean,
                        tol_std=tol_std, 
                        error_std=error_std,
                        )
        return 
    

    
    def set_voters(self, coords, sizes, widths,
                   tol_mean=1., tol_std=1., error_std=0.,):
        """
        Parameters
        ----------
        coords : array shaped (a, b) 
            Centroids of a faction voter preferences.
        
            - rows `a` = coordinate for each faction
            - columns `b' = preference dimensions. The more columns, the more preference dimensions. 
            
        sizes : array shaped (a,)
            Number of voters within each faction, with a total of `a` factions.
            Use this array to specify how many people are in each faction.
        widths : array shaped (a, b)
            The preference spread, width, or scale of the faction. These spreads
            may be multidimensional. Use columns to specify additional dimensions.         
        
        """
        seed = self._seed_voters
        rs = self._RandomState(seed, 2)
        
        coords = np.array(coords)
        sizes = np.array(sizes)
        widths = np.array(widths)
        
        voters = gaussian_preferences(coords, sizes, widths, rstate=rs)
#        tolerance = rs.normal(tol_mean, tol_std, size=voters.shape[0])
#        error = np.abs(rs.normal(0, error_std, size=voters.shape[0]))
        
        tolerance = ltruncnorm(loc=tol_mean,
                               scale=tol_std,
                               size=voters.shape[0],
                               random_state=rs)
        error = ltruncnorm(loc=tol_mean,
                            scale=tol_std,
                            size=voters.shape[0],
                            random_state=rs)
        
        self._stats.run(voters)
        
        self.voters = voters
        self.tolerance = tolerance
        self.error = error
        
        voter_args = {}
        voter_args['coords'] = coords
        voter_args['sizes'] = sizes
        voter_args['widths'] = widths
        voter_args['tol_mean'] = tol_mean
        voter_args['tol_std'] = tol_std
        voter_args['error_std'] = error_std
        voter_args['seed'] = seed
        voter_args['ndim'] = coords.shape[1]
        
        self._voter_args = voter_args      
        return 
    
    

                
    def generate_candidates(self, cnum, sdev=2, ):
        """
        Parameters
        ----------
        cnum : int
            Number of candidates for election
        sdev : float
            +- Width of standard deviations to set uniform candidate generation across population
        
        """
        seed = self._seed_candidates
        rs = self._RandomState(seed, 3)

        std = self._stats['voter.std']
        mean = self._stats['voter.mean']
        ndim = self._voter_args['ndim']

        candidates = rs.uniform(low = -sdev*std,
                                high = sdev*std,
                                size = (cnum, ndim)) + mean
        self.candidates = candidates
        c_args = {}
        c_args['cnum'] = cnum
        c_args['sdev'] = sdev
        c_args['seed'] = seed
        c_args['coords'] = candidates
        
        self._candidate_args = c_args
        return
    
    
    def add_candidates(self, candidates):
        """Add 2d array of candidates to election"""
        try:
            self.candidates = np.row_stack((self.candidates, candidates))
        except AttributeError:
            self.candidates = np.atleast_2d(candidates)
            self._candidate_args = {}
        return
    
    
    def add_median_candidate(self,):
        median = self._stats['voter.median']
        self.add_candidates(median)
        
        
    def add_faction_candidate(self, vindex):
        """
        Add a candidate lying on the centroid of a faction generated using
        self.set_voters or set_random_voters. 
        
        Parameters
        ----------
        vindex : int
            Index of faction, found in self.voter_ags['coords']
            
        """
        coords = self._voter_args['coords'][vindex]
        self.add_candidates(coords)
        return
    
        
    
    def run(self, etype=None, method=None, btype=None, 
            numwinners=1, scoremax=None, kwargs=None):
        
        """Run the election & obtain results. For ties, randomly choose winner. 
        
        Parameters
        ----------
        etype : str
            Name of election type.
            Mutually exclusive with `method` and `btype`
            Supports the following election types, for example:
                
            - 'rrv' -- Reweighted range and pure score voting
            - 'irv' -- Single tranferable vote, instant runoff.
            - 'plurality' -- Traditional plurality & Single No Transferable Vote.
            
                
        method : func
            Voting method function. Takes in argument `data` array shaped (a, b)
            for (voters, candidates) as well as additional kwargs.
            Mutually exclusive with `etype`.
            
            >>> out = method(data, numwin=self.numwinneres, **kwargs)
            
        btype : str
            Voting method's ballot type. 
            Mutually exclusive with `etype`, use with `method`.
            
            - 'rank' -- Use candidate ranking from 1 (first place) to n (last plast), with 0 for unranked.
            - 'score' -- Use candidate rating/scored method.
            - 'vote' -- Use traditional, single-selection vote. Vote for one (1), everyone else zero (0).  
        """
        seed = self._seed_election
        
        run_args = {}
        run_args['etype'] = etype
        run_args['method'] = method
        run_args['btype'] = btype
        run_args['numwinners'] = numwinners
        run_args['scoremax'] = scoremax
        run_args['kwargs'] = kwargs
        run_args['seed'] = seed
        
        if seed is None:
            seed2 = seed
        else:
            seed2 = (seed, 4)
        
        
        e = ElectionRun(self.voters, 
                        self.candidates,
                        numwinners=numwinners,
                        cnum=None,
                        error=self.error,
                        tol=self.tolerance)    
        
        e.run(etype, 
              method=method,
              btype=btype, 
              scoremax=scoremax, 
              seed=seed2,
              kwargs=kwargs)
        
        stats = metrics.ElectionStats(voters=self.voters,
                                      candidates=self.candidates,
                                      winners=e.winners,
                                      ballots=e.ballots)        
        
        ### Build dictionary of all arguments and results 
        results = {}
        results['args.candidate'] = self._candidate_args
        results['args.voter'] = self._voter_args
        results['args.election'] = run_args
        
        for key, val in self._args.items():
            newkey = 'args.user.' + key
            results[newkey] = val
        
        results['stats'] = stats.stats
        results['stats']['ties'] = e.ties
        results = utilities.flatten_dict(results, sep='.')
        self.results = results        
        
        self._result_history.append(results)
        return results
    

    def dataseries(self):
        """Retrieve pandas data series of output data"""        
        return pd.Series(self.results)
    
    
    def dataframe(self):
        """Construct data frame from results history"""
        
        series = []
        for r in self._result_history:
            series.append(pd.Series(r))
        df = pd.concat(series, axis=1).transpose()
        self._dataframe = df
        return df
    
    def save(self, name):
        self._dataframe.to_json(name)
        
        
    def rerun(**kwargs):
        """Re-run election using dataframe output"""
        
        d = kwargs.copy()
        for k in d:
            if not k.startswith('args.'):
                d.pop(k)
        
        
        e = Election()
        self.candidates = d['args.candidate.coords']

        
        
        
        
        
    
#def build_dataframe(results):
#    """Build a dataframe from a list of Election.results
#    
#    Parameters
#    -----------
#    elections : list of Election.results
#        After election has been run
#        
#    
#    """
#
#
#    a = [e.dataseries() for e in elections]
#    df = pd.concat(a, axis=1).transpose()        
#    return df
#
#        
#    
#
#








    
def plot1d(election, results, title=''):
    """Visualize election for 1-dimensional preferences
    
    Parameters
    ----------
    election : Election object
    
    results : list of ElectionResults
        Results of various election methods to compare
    """
    
    v = election.voters
    c = election.candidates

    markers = itertools.cycle(('o','v','^','<','>','s','*','+','P')) 
    
    h, edges = np.histogram(v, bins=20, density=True)
    bin_centers = .5*(edges[0:-1] + edges[1:])

    # create plot for candidate preferences
    yc = np.interp(c.ravel(), bin_centers, h, )
    
    fig, ax = plt.subplots()
    ax.plot(bin_centers, h, label='voter distribution')    
    ax.plot(c, yc, 'o', ms=10, fillstyle='none', label='candidates')
    
    # create plot for winner preferences
    for result in results:
        w = result.winners
        cw = c[w]    
        yw = np.interp(cw.ravel(), bin_centers, h, )    
#        ax.plot(cw, yw, ms=10, marker=next(markers), label=result.methodname)
        ax.plot(cw, yw, ms=6.5, marker=next(markers), label=result.methodname)
#        ax.annotate(result.methodname, (cw, yw + .01))

    ax.set_xlabel('Voter Preference')
    ax.set_ylabel('Voter Population Density')

    mean = election.stats.mean_voter
    median = election.stats.median_voter
    ymean = np.interp(mean, bin_centers, h,)
    ymedian = np.interp(median, bin_centers, h,)    

    ax.plot(mean, ymean, '+', label='mean')
    ax.plot(median, ymedian, 'x', label='median')    
    plt.legend()
    plt.grid()
    plt.title(title)
    # create plot of regrets for all possible 1-d candidates within 2 standard deviations
    arr1 = np.linspace(bin_centers[0], bin_centers[-1], 50)
    r = metrics.candidate_regrets(v, arr1[:, None])
    ax2 = ax.twinx()
    ax2.plot(arr1, r, 'r', label='Pref. Regret')
    ax2.set_ylabel('Voter Regret')
    ax2.set_ylim(0, None)
    plt.legend()
    

    
def plot_hist(output):
    """
    Plot histogram information from output from `simulate_election`
    """
    edges = output['h_edges']
    
    
    xedges = 0.5 * (edges[0:-1] + edges[1:])
    voters = output['h_voters'] 
    candidates = output['h_candidates']
    winners = output['h_winners']
    print(winners)
    plt.plot(xedges, voters, label='voters')
    plt.plot(xedges, candidates, 'o-', label='candidates')
    plt.plot(xedges, winners, 'o-', label='winners')
    plt.legend()



def plot2d(election, results, title=''):

    v = election.voters
    c = election.candidates
    markers = itertools.cycle(('o','v','^','<','>','s','*','+','P')) 
    
    h, xedges, yedges = np.histogram2d(v[:,0], v[:,1], bins=20, normed=True)
    xbin_centers = .5*(xedges[0:-1] + xedges[1:])
    ybin_centers = .5*(yedges[0:-1] + yedges[1:])
    
    fig = plt.figure(figsize=(12, 8))
#    plt.pcolormesh(xbin_centers, ybin_centers, h,)
    plt.contourf(xbin_centers, ybin_centers, h, 20)
    plt.plot(c[:,0], c[:,1], 'o', ms=10, label='candidates')
    
    for result in results:
        w = result.winners
        cw = c[w]
        plt.plot(cw[:,0], cw[:, 1],
                 ms=6.5,
                 marker=next(markers), 
                 label=result.methodname)
    
    plt.xlabel('Voter Preference 0')
    plt.ylabel('Voter Preference 1')    
    mean = election.stats.mean_voter
    median = election.stats.median_voter
    plt.plot(mean[0], mean[1], '+', label='mean')
    plt.plot(median[0], median[1], 'x', label='median')        
    plt.legend()
    
    #
        
if __name__ == '__main__':
    
    avg_errors = []
    std_errors = []
    median_errors = []
    hist_errors = []
    for seed in range(10):
        print('------------------------------------------------------------------')
        print('')
        #seed = 0
        rstate = randomstate.state
        num_voters = 300
        num_factions = 5
        num_candidates = 50
        num_winners = 1
        ndim = 1
        voters = generate_preferences(num_factions, num_voters, ndim=ndim, seed=seed)
        candidates = generate_preferences(3, num_candidates, ndim=ndim, sepfactor=3, seed=seed)
        
        e1 = ElectionPlurality(voters, candidates, numwinners=num_winners)
        e2 = ElectionScore(voters, candidates, numwinners=num_winners)
        e3 = ElectionIRV(voters, candidates, numwinners=num_winners)
        print('RUNNING PLURALITY ELECTION ------------')
        out1 = e1.run()
        print('RUNNING SCORE ELECTION ------------')
        out2 = e2.run()
        print('RUNNING IRV ELECTION ------------')
        out3 = e3.run()
        
        avg = [a['avg_error'] for a in (out1, out2, out3)]
        std = [a['std_error'] for a in (out1, out2, out3)]
        median = [a['median_error'] for a in (out1, out2, out3)]
        hist = [a['hist_error'] for a in (out1, out2, out3)]
        
        avg_errors.append(avg)
        std_errors.append(std)
        median_errors.append(median)
        hist_errors.append(hist)
#    
#np.random.seed(None)
#numvoters = 10000
#numcandidates = 40
#numwinners = 1
#
#### Create array of voter prefere5nces
#voters1 = np.random.normal(size=int(numvoters/2)) + 3
#voters2 = np.random.normal(size=int(numvoters/2)) - 2
#voters = np.append(voters1, voters2)
##voters = voters2
##np.random.seed(1)
#tol = 1
##method = score.reweighted_range
##method = irv.IRV_STV
#method = plurality.plurality
#mtype = 'score'
##mtype = 'rank'
#output = simulate_election(voters, candidates, tol, numwinners, 
#                  method=method,
#                  mtype=mtype,)
#stat_output = output_stats(output)
#plot_hist(stat_output)
#
#print_key(stat_output, 'avg_error')
#print_key(stat_output, 'std_error')
#print_key(stat_output, 'median_error')
#print_key(stat_output, 'hist_error')

#
#    output['winner_avg_preference'] = np.mean(candidates[winners])
#    output['winner_median_preference = np.median(candidates[winners])
#    winner_std_preference = np.std(candidates[winners])
#    voter_avg_preference = np.mean(voters)
#    voter_median_preference = np.median(voters)
#    voter_std_preference = np.std(voters)

    
        

#winners, ties, history = score.reweighted_range(scores, C_ratio=1, numwin=numwinners)
#winners, ties = plurality.plurality(scores, numwin=numwinners)
#winners = rcv.STV_calculator(ranks, winners=numwinners)

#
#h_voters, edges1 = np.histogram(voters, bins=20)
#h_candidates, edges2 = np.histogram(candidates, bins=20)
#h_winners, edges3 = np.histogram(candidates[winners], bins=20)
#
#
#
#print('voter avg preference = %.3f' % voter_avg_preference)
#print('voter median preference = %.3f' % voter_median_preference)
#print('voter std preference = %.3f' % voter_std_preference)
#print('winner avg preference = %.3f' % winner_avg_preference)
#print('winner median preference = %.3f' % winner_median_preference)
#print('winner std preference = %.3f' % winner_std_preference)
#print('')
#plt.figure()
#plt.plot(edges1[0:-1], h_voters / h_voters.max(), '.-', label='voters')
#plt.plot(edges2[0:-1], h_candidates / h_candidates.max(), '.-', label='candidates')
#plt.plot(edges3[0:-1], h_winners / h_winners.max(), 'o-', label='winners')
#plt.legend()