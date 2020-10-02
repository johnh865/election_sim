# -*- coding: utf-8 -*-

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

Object Data Transfer Model
--------------------------

Voters --> VoterGroup

Voters/VoterGroup --> Candidates

Voters, VoterGroup, Candidates --> Election


To construct models or benchmarks, start by creating object `Voters`. 
`Voters` may have various properties such as preference, 
voter strategy parameters, tolerance circles, etc. Define these
properties in Voters. Voters can be segregated by groups, 
and each group may have different properties. `VoterGroup` is used to 
define groups of several `Voters`. 

After defining voters, candidates may be defined using class
`Candidate`. `Candidate` definition may be dependent on the voters population,
therefore `Candidate` accepts voters as an argument. 

With the voters and candidates define, an election can be generated with
`Election`. `Election` has many subclasses which run the election. 

- `VoterBallots` takes voter and candidate information to generate honest 
  and tactical ballots.
- `eRunner` handles the running of specific types of elections.
- `ElectionResult` handles the storage of output data. 

"""
import pickle
import copy

import numpy as np
import pandas as pd
import scipy
from scipy.stats import truncnorm

from votesim import metrics
from votesim import ballot
from votesim import votesystems
from votesim import utilities
from votesim.models import vcalcs


__all__ = [
    'Voters',
    'VoterGroup',
    'Candidates',
    'Election'
           ]

# Base random seeds
VOTERS_BASE_SEED = 2
CLIMIT_BASE_SEED = 3
CANDIDATES_BASE_SEED = 4
ELECTION_BASE_SEED = 5


#import seaborn as sns

import logging
logger = logging.getLogger(__name__)




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
    if scale == 0:
        return np.ones(size) * loc
    
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
        rstate = np.random.RandomState
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
        
        
# def generate_preferences(numfactions, size, ndim=1, sepfactor=1, rstate=None):
#     """
#     Create multi-peaked gaussian distributions of preferences
    
#     Parameters
#     ----------
#     numvoters : int array of shape (a,), or int
#         Number of voter preferences to generate. If list/array, generate 
#         multiple gaussian voter preference peaks. 
#     ndim : int, default=1
#         Number of preference dimensions to consider
#     sepfactor : float
#         Scaling factor of typical distance of peak centers away from one another
#     seed : None (default) or int
#         Convert preference generation to become deterministic & pseudo-random
#         for testing purposes. 
        
#         - If None, use internal clock for seed generation
#         - If int, use this seed to generate future random variables. 
        
#     Returns
#     -------
#     out : array shaped (c, ndim)
#         Voter preferences for ndim number of dimensions.
        
#     Example
#     -------
    
#     Create a 2-dimensional preference spectrum, with 3 factions/preference 
#     peaks with:
#         - 200 voters in the 1st faction
#         - 400 voters in the 2nd faction
#         - 100 voters in the 3rd faction
    
#     >>> p = generate_voter_preference((200, 400, 100), 2)
    
#     Create a 1-dimensional preference spectrum with gaussian distribution
#     for 500 voters.
    
#     >>> p = generate_voter_preferences(500, 1)
    
#     """
#     if rstate is None:
#         rstate = randomstate.state
    
#     size1 = int(size/3)
#     numvoters = [rstate.randint(size1, size) for i in range(numfactions)]
#     new = []
#     numvoters = np.atleast_1d(numvoters)
#     for pop_subset in numvoters:

#         center = (rstate.rand(ndim) - 0.5) * sepfactor
#         scale = rstate.rayleigh(size=ndim) / 2
#         pi = rstate.normal(loc=center,
#                               scale=scale, 
#                               size=(pop_subset, ndim))
#         new.append(pi)
#     new = np.vstack(new)
#     return new
        


def _RandomState(seed, level=1):
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
    
   
class Voters(object):
    """Create simple normal distribution of voters.
    
    Parameters
    ----------
    seed : int or None
        Integer seed for pseudo-random generation. None for random numbers.
    strategy : dict
        Voter regret-to-ratings conversion strategy. 
    
    stol : float (default 1.0)
        Tolerance factor for strategy
        
    Features
    --------
    Score & ratings are constructed based on candidate coordinates
    
    
    Attributes
    ----------
    pref : array shape (a, b)
        Voter preferences, `a` number of voters, `b` number of preference dimensions

    strategy : dict
        Container for strategy options with keys
        
        tol : float
            Voter preference tolerance
        base : str
            Base honest ballot type
        tactics : list of str
            Tactic methods to apply onto ballot.
            See `votesim.ballot.TacticalBallots` for available tactics. 
        onesided : bool
            Use onesided ballot, or use full strategic ballot.
        iteration : int
            Numbers of iterations of strategy to undergo. 
    """
    def __init__(self, seed=None, strategy=None, order=1):
        self.init(seed, order=order)
        if strategy is None:
            strategy = {}
        
        self.set_strategy(**strategy)
        return
        

    @utilities.recorder.record_actions(replace=True)
    def init(self, seed, order):
        """Set pseudorandom seed."""
        self.seed = seed
        self._randomstate = _RandomState(seed, VOTERS_BASE_SEED)  
        self.order = order
        #self._randomstate2 = _RandomState(seed, CLIMIT_BASE_SEED)  
        return        
    
    
    @utilities.recorder.record_actions(replace=True)
    def set_strategy(self, tol=None, base='linear', iterations=1,
                     tactics=(), onesided=False):
        """Set voter strategy type."""
        self.strategy = {}
        self.strategy['tol'] = tol
        self.strategy['base'] = base
        self.strategy['tactics'] = tactics
        self.strategy['onesided'] = onesided
        
        if len(tactics) == 0:
            iterations = 0
            
        self.strategy['iterations'] = iterations
        
        
    @utilities.recorder.record_actions()
    def add_random(self, numvoters, ndim=1, loc=None):
        """Add random normal distribution of voters.
        
        Parameters
        ----------
        numvoters : int
            Number of voters to generate
        ndim : int
            Number of preference dimensions of population
        loc : array shaped (ndim,)
            Coordinate of voter centroid
        """
        rs = self._randomstate
        center = np.zeros(ndim)
        
        voters = rs.normal(center, size=(numvoters, ndim))
        if loc is not None:
            voters = voters + loc
            
        self._add_voters(voters)
        return
    
    
    @utilities.recorder.record_actions()
    def add_points(self, avgnum, pnum, ndim=1):
        """Add a random point with several clone voters at that point.
        
        Parameters
        ----------
        avgnum : int
            Avg. Number of voters per unique point
        pnum : int
            Number of unique points
        ndim : int
            Number of dimensions
        """
        rs = self._randomstate
        
        center = np.zeros(ndim)
        
        for i in range(pnum):
            
            # coordinate of point
            point = rs.normal(center, size=(1, ndim))
            
            # number of voters at the point
            voternum = ltruncnorm(1, 1, 1) * avgnum
            voternum = int(voternum)
            
            voters = np.ones((voternum, ndim)) * point
            self._add_voters(voters)
        return
    
    
    @utilities.recorder.record_actions()
    def add(self, pref):
        """Add arbitrary voters.
        
        Parameters
        ----------
        pref : array shape (a, b)
            Voter preferences, `a` is number of voters, `b` pref. dimensions.
        """
        self._add_voters(pref)
        pass
    
    
    
    def _add_voters(self, pref):
        """Base function for adding 2d array of candidates to election."""
        
        try:
            pref = np.row_stack((self.pref, pref))
        except AttributeError:
            pref = np.atleast_2d(pref)
            
        self.pref = pref
        self._ElectionStats = metrics.ElectionStats(voters=self)
        return
        
    
    


    def calculate_distances(self, candidates):
        """`

        
        Parameters
        ----------
        candidates : array shaped (a, b)
            Candidate preference data

        """        
        
        
        pref = self.pref
        try:
            weights = self.weights
        except AttributeError:
            weights = None
            
        distances = vcalcs.voter_distances(voters=pref,
                                           candidates=candidates,
                                           weights=weights,
                                           order=self.order)
        return distances
    
    
    def honest_ballots(self, candidates):
        """Honest ballots calculated from Candidates."""
        distances = self.calculate_distances(candidates.pref)
        b = ballot.gen_honest_ballots(distances=distances,
                                       tol=self.strategy['tol'],
                                       base=self.strategy['base'])
        return b
    
            
    
    @property
    def electionStats(self):
        return self._ElectionStats

            
    def reset(self):
        """Reset method records. Delete voter preferences and records."""
        try:
            self._method_records.reset()
        except AttributeError:
            pass
        try:
            del self.pref
        except AttributeError:
            pass
        return
    

class VoterGroup(object):
    """Group together multiple voter objects & interact with candidates."""
    def __init__(self, voters_list):        
        try:
            iter(voters_list)
        except Exception:
            voters_list = [voters_list]            
        self.group = voters_list        
        orders = np.array([v.order for v in self.group])

        if not np.all(orders == orders[0]):
            raise ValueError('Order of voters in group must all be same.')
        self.order = orders[0]
        return
    
    
    @utilities.lazy_property
    def pref(self):
        vlist = [v.pref for v in self.group]
        return np.vstack(vlist)
    
    
    @utilities.lazy_property
    def electionStats(self):
        return metrics.ElectionStats(voters=self)
    
    
    def reset(self):
        for voter in self.group:
            voter.reset()
        
    
    
    def __getitem__(self, key):
        return self.group[key]
    
    
def voter_group(vlist):
    """Group together multiple Voters."""
    if hasattr(vlist, 'group'):
        return vlist
    else:
        return VoterGroup(vlist)
        

    
class Candidates(object):
    """
    Create candidates for spatial model
    
    Parameters
    -----------
    voters : `Voters` or `VoterGroup`
        Voters to draw population data. 
    
    Attributes
    ----------
    pref : array shape (a, b)
        Voter preferences, `a` number of candidates, 
        `b` number of preference dimensions    
    """
    def __init__(self, voters: Voters, seed=None):
        self._method_records = utilities.recorder.RecordActionCache()
        
        if not hasattr(voters, '__len__'):     
            voters = [voters]
            
        self.voters = voter_group(voters)
        self.set_seed(seed)
        return    
    
    
    @utilities.recorder.record_actions()
    def set_seed(self, seed):
        """ Set pseudorandom seed """
        self._seed = (seed, CANDIDATES_BASE_SEED)
        self._randomstate = _RandomState(*self._seed)
        return
    
    
    def _add_candidates(self, candidates):
        """Base function for adding 2d array of candidates to election"""
        candidates = np.array(candidates)
        assert candidates.ndim == 2, 'candidates array must have ndim=2'
        
        try:
            candidates = np.row_stack((self.candidates, candidates))
        except AttributeError:
            candidates = np.atleast_2d(candidates)
        
        cdim = candidates.shape[1]
        vdim = self.voters.pref.shape[1]
        
        condition = cdim == vdim
        s = ('dim[1] of candidates (%s) '
             'must be same as dim[1] (%s) of self.voters' % (cdim, vdim))
            
        assert condition, s
        self.pref = candidates
        return
       
    
    def reset(self):
        """Reset candidates for a given Voters.
        Delete candidate preferences and records"""
        try:
            self._method_records.reset()
        except AttributeError:
            pass
        try:
            del self.pref
        except AttributeError:
            pass
        return
    
    
    @utilities.recorder.record_actions()
    def add_random(self, cnum, sdev=2):
        """
        Add random candidates, uniformly distributed.
        
        Parameters
        ----------
        cnum : int
            Number of candidates for election
        sdev : float
            +- Width of standard deviations to set uniform candidate
            generation across voter population
        
        """
        rs = self._randomstate
        std = self.voters.electionStats.voter.pref_std
        mean = self.voters.electionStats.voter.pref_mean
        ndim = std.shape[0]
        
        candidates = rs.uniform(low = -sdev*std,
                                high = sdev*std,
                                size = (cnum, ndim)) + mean
        self._add_candidates(candidates)
        return
    
    
    @utilities.recorder.record_actions()
    def add(self, candidates):
        """Add 2d array of candidates to election, record actions
        
        Parameters
        ----------------
        candidates : array shape (a, n)
            Candidate preference coordinates.
            
            - a = number of candidates
            - n = number of preference dimensions
        """
        self._add_candidates(candidates)
        return
        
    
    @utilities.recorder.record_actions()
    def add_median(self,):
        """Add candidate located at voter median coordinate"""
        median = self._stats['voter.median']
        self._add_candidates(median)
        
        
    @utilities.recorder.record_actions()
    def add_faction(self, vindex):
        """
        Add a candidate lying on the centroid of a faction generated using
        Voters.add_faction.
        
        Parameters
        ----------
        vindex : int
            Index of faction, found in self.voter_ags['coords']
            
        """
        coords = self.voters.fcoords[vindex]
        self._add_candidates(coords)
        return
    
    
    # def get_ballots(self, etype):
    #     return self.voters.tactical_ballots(etype)
    
    
class VoterBallots(object):
    """
    Generate ballots from voter and candidate data.
    
    Parameters
    ----------
    voters_list : list of Voter or VoterGroup
        Voters of election
    candidates : Candidates
        Candidates of election
        
    """
    def __init__(self, voters_list, candidates):
        self.candidates = candidates
        self.group = voter_group(voters_list).group


    @utilities.lazy_property
    def honest_ballots(self) -> ballot.CombineBallots:
        """Combined honest ballots for all voters in all groups."""
        blist = [v.honest_ballots(self.candidates) for v in self.group]
        new = ballot.CombineBallots(blist)
        return new
    
    
    def ballots(self, etype, ballots=None) -> ballot.TacticalBallots: 
        """Generate ballots according specified voter strategy.
        
        One-sided index information for `self.index_dict` is also constructed 
        when tactical ballots are constructed. 
        
        Parameters
        ----------
        etype : str
            Election type
        ballots : ballot subclass
            Initial ballots
        """
        indices = self.honest_ballots.children_indices
        maxiter = max(v.strategy['iterations'] for  v in self.group)
        
        if ballots is None:
            b0 = self.honest_ballots
        else:
            b0 = ballots
        if maxiter == 0:
            return b0
        
        # Iteratively generate ballots if multiple iterations desired. 
        for ii in range(maxiter):
            
            b = ballot.TacticalBallots(etype, ballots=b0)            
            
            # Set tactics for each group
            for jj, vindex in enumerate(indices):
                voters = self.group[jj]
                strategy = voters.strategy
                iterations = strategy['iterations']
                
                if ii < iterations:
                    b.set(tactics=strategy['tactics'],
                          onesided=strategy['onesided'],
                          index=indices[jj]
                          )

                # Record group index locations for one-sided tactics
                if ii == iterations - 1:
                    if strategy['onesided'] == True:
                        name = str(ii) + '-underdog'
                        self.index_dict[name] = b.iloc_bool_underdog
                        name = str(ii) + '-topdog'
                        self.index_dict[name] = b.iloc_bool_topdog

            # To perform next iteration, set the base ballot to the newly
            # constructed tactical ballots
            b0 = b
        return b
    
    
    @utilities.lazy_property
    def index_dict(self):
        """dict : Index locations of voters for each group. 
        
        If one-sided tactical ballots are generated, index locations for
        '-topdog' and '-underdog' voters are also included."""
        indices = self.honest_ballots.children_indices
        index_dict = {}
        for ii, index in enumerate(indices):
            index_dict[str(ii)] = index
        #self._index_dict = index_dict
        return index_dict
    
    
    def reset(self):
        utilities.clean_lazy_properties(self)
    
        
    
    @property
    def distances(self):
        """(a, b) array: `a` Voter preference distances from `b` candidates."""
        return self.honest_ballots.distances
    
    
    def __getitem__(self, key):
        return self.group[key]
              
        

class Election(object):
    """
    Run an Election with Voters and Candidates
    
    Parameters
    ------------
    voters : None, Voters, VoterGroup, or list of Voters
        Voters object specifying the voter preferences and behavior.
    candidate : None or Candidates
        Candidates object specifying candidate preferences
    seed : int or None
        Seed for pseudo-random number generation
    numwinners : int >= 1
        Number of winners for the election
    scoremax : int
        Maximum score for ballot generation
    name : str
        Name of election model, used to identify different benchmark models.
    save_args : bool (default True)
    
        - If True, save all parameters input into method calls. These 
          parameters can be used to regenerate specific elections. 
        - If False, only save parameters input into `self.user_data`.    
        
    Attributes
    ----------
    result : ElectionResult
        Results storage for Election.
    vballots : VoterBallots
        VoterBallot data
    
    """
    def __init__(self,
                 voters: VoterGroup=None,
                 candidates: Candidates=None,
                 seed=None,
                 numwinners=1,
                 scoremax=5,
                 name = '',
                 save_args=True):
        
        self._method_records = utilities.recorder.RecordActionCache()
        #self._result_history = []
        self.voters = None
        self.candidates = None
        self.save_args = save_args

        self.init(seed, numwinners, scoremax, name)
        self.set_models(voters, candidates)
        self.result = ElectionResult(self)
        return
    
    
    @utilities.recorder.record_actions(replace=True)
    def init(self, seed, numwinners, scoremax, name):
        """Initialize some election properties"""
        self._set_seed(seed)
        self.numwinners = numwinners
        self.scoremax = scoremax
        self.name = name
        
        
        
        return
    
    
    def set_models(self, voters=None, candidates=None):
        """Set new voter or candidate model.
        
        Parameters
        ----------
        voters : Voters or None
            New voters object
        candidates : Candidates or None
            New candidates object
        """
        if voters is not None:
            self.voters = voter_group(voters)
            self.electionStats = self.voters.electionStats

        if candidates is not None:
            self.candidates = candidates
            self.electionStats.set_data(candidates=candidates)
            # self.electionStats.set_data(candidates=self.candidates.pref,)     
            if voters is not None:
                self.vballots = VoterBallots(self.voters, self.candidates)
        return
    
    
    def _set_seed(self, seed):
        """ Set pseudorandom seed """
        if seed is None:
            self._seed = None
            self._randomstate = _RandomState(None)
        else:
            self._seed = (seed, ELECTION_BASE_SEED)
            self._randomstate = _RandomState(*self._seed)
        return
    
    
    #@utilities.recorder.record_actions(replace=True)
    def user_data(self, d=None, **kwargs):
        """Record any additional data the user wishes to record.
        
        Parameters
        ----------
        **d : dict 
            Write any keys and associated data here 
        """
        udict = {}
        udict.update(kwargs)
        
        if d is not None:
            # d is supposed to be a dictionary. Try to update our dict with it            
            try:
                udict.update(d)
                
            # Maybe the user is trying to create a parameter `d`
            except TypeError:
                udict['d'] = d
        
        self._user_data = udict
        return
    
    
    
    def reset(self):
        """Delete election data for the current run --
        voter preferences, candidate preferences, and ballots,
        Clear the kind of data that can be regenerated if desired.
        
        Do not clear statistics.
        """
        self.voters.reset()
        self.candidates.reset()
        
        def delete(a):
            try:
                delattr(self, a)
            except AttributeError:
                pass
                        

        delete('winners')
        delete('ties')
        delete('output')
        delete('vballots')
        return
    
    
    # def _generate_ballots(self):
    #     """Construct ballots"""
    #     b = self.voters.
        
    #     # c = self.candidates.candidates
    #     # ratings = self.voters.calc_ratings(c)
        
        
        
    #     # self.ballot_gen = BallotGenerator(ratings, scoremax=self.scoremax)
        
    #     # ranks = votesystems.tools.score2rank(ratings)
    #     # scores = np.round(ratings * self.scoremax)
    #     # votes = votesystems.tools.getplurality(ranks=ranks)
        
    #     # self.ranks = ranks
    #     # self.scores = scores
    #     # self.ratings = ratings
    #     # self.votes = votes
        
    #     # logger.debug('rank data = \n%s', ranks)
    #     # logger.debug('scores data = \n%s', scores)
    #     # logger.debug('ratings data = \n%s', ratings)
    #     # logger.debug('votes data = \n%s', votes)
    #     return b
    
    
    @utilities.recorder.record_actions(replace=True)
    def run(self, etype=None, method=None,
            btype=None,  kwargs=None):
        """Run the election using `votesystems.eRunner`."""
        
        logger.debug('Running %s, %s, %s', etype, method, btype)
        
        ballots = self.vballots.ballots(etype=etype)
        runner = ballots.run(etype=etype, 
                             rstate=self._randomstate,
                             numwinners=self.numwinners)
        result = self.result.update(runner)
        self.tactical_ballots = ballots

        # runner = self.ballot_gen.run(etype=etype,
        #                           rstate=self._randomstate,
        #                           numwinners=self.numwinners)
        
        # runner = votesystems.eRunner(etype=etype, 
        #                             method=method,
        #                             btype=btype,
        #                             rstate=self._randomstate,
        #                            #seed=self._seed,
        #                             kwargs=kwargs,
        #                             numwinners=self.numwinners,
        #                             scores=self.scores,
        #                             ranks=self.ranks,
        #                             votes=self.votes,
        #                             ratings=self.ratings)
        
        # self.btype = runner.btype
        # self.winners = runner.winners
        # self.ties = runner.ties
        # self.output = runner.output
        # self.ballots = runner.ballots
        
        # self.get_results()
        return result
    

    
    
    # def get_results(self) -> dict:
    #     """Retrieve election statistics and post-process calculations"""
        
    #     stats = self.electionStats
    #     stats.set_data(
    #                    winners=self.winners,
    #                    ballots=self.ballots
    #                    )
        

    #     ### Build dictionary of all arguments and results 
    #     results = {}        
    #     results.update(self.get_parameters())
        
    #     results['output'] = stats.get_dict()
    #     results = utilities.misc.flatten_dict(results, sep='.')
    #     self.results = results        
        
    #     self._result_history.append(results)
    #     return results        
    
    
    # def get_parameter_keys(self):
    #     """Retrieve election input parameter keys"""
    #     return list(self.get_parameters().keys())
    
    
    # def get_parameters(self):
    #     """Retrieve election input parameters"""
    #     params = {}
    #     crecord = self.candidates._method_records.dict
        
    #     vrecords = []
    #     for v in self.voters.group:
    #         vrecords.append(v._method_records.dict)
        
    #     #vrecord = self.voters._method_records.dict
        
    #     erecord = self._method_records.dict

    #     # Determine if user data exists. If not, save default save_args    
    #     save_args = self.save_args
    #     try:
    #         userdata = self._user_data
    #         if len(userdata)  == 0:
    #             save_args = True
    #     except AttributeError:
    #         save_args = True
    #         userdata = {}

    #     # Add user data to params
    #     for key, value in userdata.items():
    #         newkey = 'args.user.' + key
    #         params[newkey] = value
        
    #     # Save etype and name in special parameters
    #     for key in erecord:
    #         if 'run.etype' in key:
    #             params['args.etype'] = erecord[key]
    #         elif '.init.name' in key:
    #             params['args.name'] = erecord[key]
            
    #     # Save all method call arguments
    #     if self.save_args or save_args:   
    #         params['args.candidate'] = crecord
    #         for ii, vrecord in enumerate(vrecords):
    #             params['args.voter-%s' % ii] = vrecord
    #         params['args.election'] = erecord    
            
    #     params = utilities.misc.flatten_dict(params, sep='.')
    #     return params
    
    
    # def get_output_docs(self):
    #     """Retrieve output documentation"""
    #     return self.electionStats.get_docs()

    
    # @property
    # def electionData(self) -> metrics.ElectionData:
    #     """model election data"""
    #     return self.voters.electionStats.electionData
    
    
    # @property
    # def electionStats(self) -> metrics.ElectionStats:
    #     return self.voters.electionStats


    # def dataseries(self, index=None):
    #     """Retrieve pandas data series of output data"""  
    #     if index is None:
    #         return pd.Series(self.results)
    #     else:
    #         return pd.Series(self._result_history[index])
    
    
    # def dataframe(self):
    #     """Construct data frame from results history"""
        
    #     series = []
    #     for r in self._result_history:
    #         series.append(pd.Series(r))
    #     df = pd.concat(series, axis=1).transpose()
    #     self._dataframe = df.infer_objects()
    #     return df
    
    
    # def save(self, name, reset=True):
    #     """Pickle election data
        
    #     Parameters
    #     ----------
    #     name : str
    #         Name of new pickle file to dump Election ito
    #     reset : bool
    #         If True (default), delete election data that can be regenerated.
    #     """
    #     if reset:
    #         self.reset()
            
    #     with open(name, 'wb') as file1:
    #         pickle.dump(self, file1)
    #     return
        
        
#        
#    def load_json(self, name):
#        df = pd.read_json(name)
#        rows, cols = df.shape
#        self._dataframe  = df
#        for ii in range(rows):
#            s = df.loc[ii]
#            self._result_history.append(dict(s))
#        return
        
        
    def rerun(self, d):
        """Re-run an election found in dataframe. Find the election 
        data from the dataframe index
        
        Parameters
        ----------
        index : int or None
            Election index from self._dataframe
        d : dict or None
            Dictionary or Series of election data, generated from self.dataseries.
            
        Returns
        -------
        out : Election
            Newly constructed election object with re-run parameters. 
        """
        series = d        
        
        def filterdict(d, kfilter):
            new = {}
            num = len(kfilter)
            for k, v in d.items():
                if k.startswith(kfilter):
                    newkey = k[num :]
                    new[newkey] = v
            return new

        
        filter_key = 'args.candidate.'
        c_dict = filterdict(series, filter_key)
        
        filter_key = 'args.election.'
        e_dict = filterdict(series, filter_key)
        
        vnum = len(self.voters.group)
        new_voters = []
        for ii in range(vnum):
            filter_key = 'args.voter-%s.' % ii
            v_dict = filterdict(series, filter_key)            
            v = type(self.voters.group[ii])()        

            #v = type(self.voters)()
            v._method_records.reset()
            v._method_records.run_dict(v_dict, v)
            new_voters.append(v)
                
        c = type(self.candidates)(voters=new_voters)
        c._method_records.reset()
        c._method_records.run_dict(c_dict, c)
        
        enew = Election(voters=v, candidates=c)
        enew._method_records.run_dict(e_dict, enew)
        return enew
    
    
    def copy(self):
        """Copy election"""
        return copy.deepcopy(self)
       

    def save(self, name, reset=True):
        """Pickle election data
        
        Parameters
        ----------
        name : str
            Name of new pickle file to dump Election ito
        reset : bool
            If True (default), delete election data that can be regenerated.
        """
        if reset:
            self.reset()
            
        with open(name, 'wb') as file1:
            pickle.dump(self, file1)
        return
    
    
    def dataseries(self, index=None):
        """Retrieve pandas data series of output data."""  
        return self.result.dataseries(index=index)
    
    
    def dataframe(self):
        """Construct data frame from results history."""
        return self.result.dataframe()
        


class ElectionResult(object):
    """
    Store Election result output. Generated as attribute of Election.
    
    Parameters
    ----------
    e : Election
        Election to extract results from.
        
    Attributes
    ----------
    runner : :class:`~votesim.votesystems.voterunner.eRunner`
        Output from election running class for the last run election. 
    results : dict
        Results of last run election
        
        
    Output Specification
    --------------------
    For each election output keys are generated as dataframes or dataseries.
    
    - Voter parameters are specified as `args.voter-vnum.a.func.argname`
    
     - `vnum` = Voter group number
     - `a` = Method call number (a method could be called multiple times.)
     - `func` = Name of the called method
     - `argname` = Name of the set parameter for the method. 
     
    - Candidate parameters are specified as `args.candidate.a.func.arg`
    - User parameters are specified as `args.user.name`
     - `name` is the user's inputted parameter name
     
    
    """
    def __init__(self, e: Election):
        self.election = e
        self.save_args = e.save_args
        self._result_history = []
        pass
    
    
    def update(self, runner: votesystems.eRunner):
        """Get election results."""
        self.runner = runner   
        self.winners = runner.winners
        self.ties = runner.ties
        self.ballots = runner.ballots
        
        return self._get_results()    

    
    def _get_results(self) -> dict:
        """Retrieve election statistics and post-process calculations."""
        stats = self._electionStats
        stats.set_data(election=self.election)
        
        ### Build dictionary of all arguments and results 
        results = {}        
        results.update(self._get_parameters())
        
        results['output'] = stats.get_dict()
        results = utilities.misc.flatten_dict(results, sep='.')
        self.output = results        
        
        self._result_history.append(results)
        return results        
    
    
    def _get_parameter_keys(self) -> list:
        """Retrieve election input parameter keys."""
        return list(self._get_parameters().keys())
    
    
    def _get_parameters(self) -> dict:
        """Retrieve election input parameters."""
        params = {}
        candidates = self.election.candidates
        voters = self.election.voters
        election = self.election
        
        # get candidate parameters
        crecord = candidates._method_records.dict
        
        # get voter parameters
        vrecords = []
        for v in voters.group:
            vrecords.append(v._method_records.dict)
        
        # get election parametesr
        erecord = election._method_records.dict
        
        # Retrieve user data
        # Determine if user data exists. If not, save default save_args    
        save_args = self.save_args
        try:
            userdata = self.election._user_data
            if len(userdata)  == 0:
                save_args = True
        except AttributeError:
            save_args = True
            userdata = {}

        # Add user data to params
        for key, value in userdata.items():
            newkey = 'args.user.' + key
            params[newkey] = value
        
        # Save etype and name in special parameters
        for key in erecord:
            if 'run.etype' in key:
                params['args.etype'] = erecord[key]
            elif '.init.name' in key:
                params['args.name'] = erecord[key]
            
        # Save all method call arguments
        if self.save_args or save_args:   
            params['args.candidate'] = crecord
            for ii, vrecord in enumerate(vrecords):
                params['args.voter-%s' % ii] = vrecord
            params['args.election'] = erecord    
            
        params = utilities.misc.flatten_dict(params, sep='.')
        return params
    
    
    @utilities.lazy_property
    def output_docs(self) -> dict:
        """Retrieve output documentation."""
        return self.electionStats.get_docs()

    
    @property
    def _electionData(self) -> metrics.ElectionData:
        """model election data."""
        return self.election.electionStats.electionData
    
    
    @property
    def _electionStats(self) -> metrics.ElectionStats:
        return self.election.electionStats


    def dataseries(self, index=None):
        """Retrieve pandas data series of output data."""  
        if index is None:
            return pd.Series(self.results)
        else:
            return pd.Series(self._result_history[index])
    
    
    def dataframe(self):
        """Construct data frame from results history."""
        
        series = []
        for r in self._result_history:
            series.append(pd.Series(r))
        df = pd.concat(series, axis=1).transpose()
        self._dataframe = df.infer_objects()
        return df

        
