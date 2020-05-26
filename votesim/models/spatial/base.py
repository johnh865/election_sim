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


"""
import itertools
import pickle
import copy
from functools import wraps

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy.interpolate import NearestNDInterpolator
from scipy.stats import truncnorm

from votesim import metrics
from votesim import votesystems
from votesim import utilities
from votesim.models import vcalcs

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
    for pop_subset in numvoters:

        center = (rstate.rand(ndim) - 0.5) * sepfactor
        scale = rstate.rayleigh(size=ndim) / 2
        pi = rstate.normal(loc=center,
                              scale=scale, 
                              size=(pop_subset, ndim))
        new.append(pi)
    new = np.vstack(new)
    return new
        


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
    
   
class SimpleVoters(object):
    """
    Create simple normal distribution of voters. 
    
    Parameters
    ----------
    seed : int or None
        Integer seed for pseudo-random generation. None for random numbers.
    strategy : str
        Voter regret-to-ratings conversion strategy. Options are
        
        - 'candidate' -- Tolerance defiend by candidates
        - 'voter' -- Tolerance defined by voter population std deviation
    
    stol : float (default 1.0)
        Tolerance factor for strategy
        
    
    Features
    --------
    Score & ratings are constructed based on candidate coordinates
    
    
    Attributes
    ----------
    voters : array shape (a, b)
        Voter preferences, `a` number of voters, `b` number of preference dimensions
    ratings : array shape (a, c)
        Voter ratings for `c` candidates, calculated using `self.calc_ratings`
    distances : array shape (a, c)
        Voter distances for `c` candidates, calculated using `self.calc_ratings`
    
    """
    def __init__(self, seed=None, strategy='candidate', stol=1.0):
        self.set_seed(seed)
        self.set_strategy(strategy=strategy, stol=stol)
        return
    
    
    # @utilities.recorder.record_actions(replace=True)
    # def init(self, seed):
    #     """Initialize some election properties"""
    #     self.set_seed(seed)

    #     return
        

    @utilities.recorder.record_actions(replace=True)
    def set_seed(self, seed):
        """ Set pseudorandom seed """
        self.seed = seed
        self._randomstate = _RandomState(seed, VOTERS_BASE_SEED)  
        #self._randomstate2 = _RandomState(seed, CLIMIT_BASE_SEED)  
        return        
    
    
    @utilities.recorder.record_actions(replace=True)
    def set_strategy(self, strategy, stol):
        self.stol = stol
        self.strategy = strategy        
        
        
    @utilities.recorder.record_actions()
    def add_random(self, numvoters, ndim=1, loc=None):
        """Add random normal distribution of voters
        
        Parameters
        -----------
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
        """Add a random point with several clone voters at that point
        
        Parameters
        -----------
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
    def add(self, voters):
        self._add_voters(voters)
        pass
    
    
    
    def _add_voters(self, voters):
        """Base function for adding 2d array of candidates to election"""
        
        try:
            voters = np.row_stack((self.voters, voters))
        except AttributeError:
            voters = np.atleast_2d(voters)
            
        self.voters = voters
        self._ElectionStats = metrics.ElectionStats(voters=voters, order=1)
        return
        
    
    
    def calc_ratings(self, candidates):
        """`
        Calculate preference distances & candidate ratings for a given set 
        of candidates.
        
        Parameters
        ----------
        candidates : array shaped (a, b)
            Candidate preference data
        
        Returns
        -------
        out : array shaped (c, a)
            Voter ratings for each candidate
            
        """
        voters = self.voters
        
        distances = vcalcs.voter_distances(voters, candidates)
        ratings = vcalcs.voter_scores_by_tolerance(
                                                   voters,
                                                   candidates,
                                                   distances=distances,
                                                   tol=self.stol,
                                                   cnum=None,
                                                   strategy=self.strategy,
                                                   )
        self.ratings = ratings
        self.distances = distances
        return ratings


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
            del self.voters
        except AttributeError:
            pass
        return
    

class VotersGroup(object):
    """Group together multiple voter objects"""
    def __init__(self, group = None):
        raise NotImplementedError('Needs lots of work')
        # TODO : Need to finsh this
        if group is None:
            group = []
            
        self.group = group
        raise NotImplementedError()
        return
    
    
    def calc_ratings(self, candidates):
        out = []
        for voters in self.group:
            ratings = voters.calc_ratings(candidates)
            out.append(ratings)
        return np.vstack(out)
    
    
    @property
    def voters(self):
        vlist = [v.voters for v in self.group]
        return np.vstack(vlist)
    
    
    def ElectionStats(self):
        raise NotImplementedError()
        for voters in self.group:
            estats = metrics.ElectionStats()
    

    


    
class Candidates(object):
    """
    Create candidates for spatial model
    
    Parameters
    -----------
    voters : `Voters` from votesim.voters
    """
    def __init__(self, voters : SimpleVoters, seed=None):
        self._method_records = utilities.recorder.RecordActionCache()
        self.voters = voters
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
        vdim = self.voters.voters.shape[1]
        
        condition = cdim == vdim
        s = ('dim[1] of candidates (%s) '
             'must be same as dim[1] (%s) of self.voters' % (cdim, vdim))
            
        assert condition, s
            
        self.candidates = candidates
        return
    
    
    def reset(self):
        """Reset candidates for a given Voters.
        Delete candidate preferences and records"""
        try:
            self._method_records.reset()
        except AttributeError:
            pass
        try:
            del self.candidates
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
    
        


class Election(object):
    """
    Run an Election with Voters and Candidates
    
    Parameters
    ------------
    voters : None, SimpleVoters, or Voters
        Voters object specifying the voter preferences and behavior.
    candidate : None or Canidates
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
    
    
    """
    def __init__(self,
                 voters:SimpleVoters = None,
                 candidates:Candidates = None,
                 seed=None,
                 numwinners=1,
                 scoremax=5,
                 name = '',
                 save_args=True):
        
        self._method_records = utilities.recorder.RecordActionCache()
        self._result_history = []
        self.voters = voters
        self.candidates = candidates
        self.winners = None
        self.ballots = None
        
        self.save_args = save_args

        self.init(seed, numwinners, scoremax, name)
        
        if voters is not None and candidates is not None:
            self.set_models(voters, candidates)
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
        """Set new voter or candidate model
        
        Parameters
        ------------
        voters : SimpleVoters or None
            New voters object
        candidates : Candidates or None
            New candidates object
        """
        if voters is not None:
            self.voters = voters
        if candidates is not None:
            self.candidates = candidates

            stats = self.electionStats
            stats.set_data(candidates=self.candidates.candidates,)      
            
        self._generate_ballots()
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
    
    
    @utilities.recorder.record_actions(replace=True)
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
                        
        delete('ranks')
        delete('scores')
        delete('ratings')
        delete('votes')
        delete('btype')
        delete('winners')
        delete('ties')
        delete('output')
        delete('ballots')
        return
    
    
    def _generate_ballots(self):
        """Construct ballots"""
        c = self.candidates.candidates
        ratings = self.voters.calc_ratings(c)
        ranks = votesystems.tools.score2rank(ratings)
        scores = np.round(ratings * self.scoremax)
        votes = votesystems.tools.getplurality(ranks=ranks)
        
        self.ranks = ranks
        self.scores = scores
        self.ratings = ratings
        self.votes = votes
        
        logger.debug('rank data = \n%s', ranks)
        logger.debug('scores data = \n%s', scores)
        logger.debug('ratings data = \n%s', ratings)
        logger.debug('votes data = \n%s', votes)
        return
    
    
    @utilities.recorder.record_actions(replace=True)
    def run(self, etype=None, method=None,
            btype=None,  kwargs=None):
        """Run the election using `votesystems.eRunner`."""
        
        logger.debug('Running %s, %s, %s', etype, method, btype)
        runner = votesystems.eRunner(etype=etype, 
                                    method=method,
                                    btype=btype,
                                    seed=self._seed,
                                    kwargs=kwargs,
                                    numwinners=self.numwinners,
                                    scores=self.scores,
                                    ranks=self.ranks,
                                    votes=self.votes,
                                    ratings=self.ratings)
        
        self.btype = runner.btype
        self.winners = runner.winners
        self.ties = runner.ties
        self.output = runner.output
        self.ballots = runner.ballots
        
        self.get_results()
        return 
    
    
    @property
    def electionData(self) -> metrics.ElectionData:
        """model election data"""
        return self.voters.electionData
    
    
    @property
    def electionStats(self) -> metrics.ElectionStats:
        return self.voters.electionStats
    
    
    def get_results(self) -> dict:
        """Retrieve election statistics and post-process calculations"""
        
        stats = self.electionStats
        stats.set_data(
                       #candidates=self.candidates.candidates,
                       winners=self.winners,
                       ballots=self.ballots
                       )
        

        ### Build dictionary of all arguments and results 
        results = {}        
        results.update(self.get_parameters())
        
        results['output'] = stats.get_dict()
        results = utilities.misc.flatten_dict(results, sep='.')
        self.results = results        
        
        self._result_history.append(results)
        return results        
    
    
    def get_parameter_keys(self):
        """Retrieve election input parameter keys"""
        return list(self.get_parameters().keys())
    
    
    def get_parameters(self):
        """Retrieve election input parameters"""
        params = {}
        crecord = self.candidates._method_records.dict
        vrecord = self.voters._method_records.dict
        erecord = self._method_records.dict

        # Determine if user data exists. If not, save default save_args    
        save_args = self.save_args
        try:
            userdata = self._user_data
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
            params['args.voter'] = vrecord
            params['args.election'] = erecord    
            
        params = utilities.misc.flatten_dict(params, sep='.')
        return params
    
    
    def get_output_docs(self):
        """Retrieve output documentation"""
        return self.electionStats.get_docs()


    def dataseries(self, index=None):
        """Retrieve pandas data series of output data"""  
        if index is None:
            return pd.Series(self.results)
        else:
            return pd.Series(self._result_history[index])
    
    
    def dataframe(self):
        """Construct data frame from results history"""
        
        series = []
        for r in self._result_history:
            series.append(pd.Series(r))
        df = pd.concat(series, axis=1).transpose()
        self._dataframe = df.infer_objects()
        return df
    
    
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
                            
        filter_key = 'args.voter.'
        v_dict = filterdict(series, filter_key)
        
        filter_key = 'args.candidate.'
        c_dict = filterdict(series, filter_key)
        
        filter_key = 'args.election.'
        e_dict = filterdict(series, filter_key)
        
        v = type(self.voters)()
        v._method_records.reset()
        v._method_records.run_dict(v_dict, v)
                
        c = type(self.candidates)(voters=v)
        c._method_records.reset()
        c._method_records.run_dict(c_dict, c)
        
        enew = Election(voters=v, candidates=c,)
        enew._method_records.run_dict(e_dict, enew)
        return enew
    
    
    def copy(self):
        """Copy election"""
        return copy.deepcopy(self)
       
        
        
        
        
class ___OLD_Election(object):
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
 
class __Voters(object):
    """
    Create voters with preferneces. Each voter has the following properties:
    
    Voter Properties
    ----------------
    - Preference coordinates, n-dimensional
        The voter's preference location in spatial space. 
        
    - Preference tolerance, n-dimensional
        How tolerant the voter is to other preferences in spatial space. 
        
    - Preference weight, n-dimensional
        How much the voter cares about a particular issue.
        
    - Error
        The likely amount of error the voter will make when estimating
        preference distance from themselves to a candidate. 
        
    - Candidate Limit
        The max number of candidates a voters is willing to consider - on the 
        assumption that every voter has limited mental & research resources.
        
        
    Voter Distribution Properties
    ------------------------------
    Distributions of voters shall be created using multiple
    normal distributions. 
    
    For each voter property, there may be
    
    - mean -- The mean value of a property's distribution centroid. 
    - std -- The standard deviation of a distribution's centroid.
    - width -- The mean value of a distribution's dispersion or width
    - width_std -- The standard deviation of a distribution's dispersion or width.
    
    
    Attributes
    ---------
    voters : array shape (a, b)
        Voter preference for `a` voter num & `b` dimensions num. 
    tolerance : array shape (a, b)
        Voter tolerance for `a` voter num  and `b` dimensions num.
    error : array shape (a,)
        Voter error for `a` voters.
    weight : array shape (a, b)
        Voter dimension weights for `a` voters and `b` dimensions. 
        
    
    """
    def __init__(self, seed=None):
        self._method_records = utilities.RecordActionCache()
        self.set_seed(seed)
        

    @utilities.recorder.record_actions()
    def set_seed(self, seed):
        """ Set pseudorandom seed """
        self.seed = seed
        self._randomstate = _RandomState(seed, VOTERS_BASE_SEED)  
        self._randomstate2 = _RandomState(seed, CLIMIT_BASE_SEED)  
        return
        
    
    @utilities.recorder.record_actions()
    def add_faction(self,
                    coord,
                    size, 
                    width, 
                    tol_mean,
                    tol_width, 
                    error_width=0.0,
                    weight_mean=1.0,
                    weight_width=0.0):
        """Add a faction of normally distributed voters
        
        Parameters
        ----------
        coord : array shape (a,)
            Faction centroid preference coordinates
 
        sizes : int
            Number of voters within each faction, 
            
        width : float or array shape (a,)
            The preference spread, width, or scale of the faction. These spreads
            may be multidimensional. Use columns to specify additional dimensions.    
            
        
            
        """
   
        p, t, e, w, c = self._create_distribution(
                                               coord, size, width, 
                                               tol_mean, 
                                               tol_width,
                                               error_width,
                                               weight_mean,
                                               weight_width
                                               )
        
        try:
            self.voters = np.row_stack((self.voters, p))
            self.tolerance = np.row_stack((self.tolerance, t))
            self.error = np.append(self.error, e)
            self.weight = np.row_stack((self.weight, w))
            self.fcoords = np.row_stack((self.fcoords, coord))
            self.climit = np.append(self.climit, c)
            
        except AttributeError:
            self.voters = np.atleast_2d(p)
            self.tolerance = t
            self.error = e
            self.weight = np.atleast_2d(w)
            self.fcoords = np.atleast_2d(coord)
            self.climit = c
            
            
        return
    
    
    
    
    def _create_distribution(self,
                             coord, size, width, 
                             tol_mean, 
                             tol_std, 
                             error_std=0.0,
                             weight_mean=1.0, 
                             weight_std=1.0,
                             cnum_mean=np.nan,
                             cnum_std=1.0
                             ):
        """Perform calculations for add_faction"""
        
        rs = self._randomstate
        coord = np.array(coord)
        ndim = len(coord)
        
        preferences = rs.normal(
                               loc=coord,
                               scale=width,
                               size=(size, ndim),
                               )
        tolerance = ltruncnorm(
                               loc=tol_mean,
                               scale=tol_std,
                               size=size,
                               random_state=rs,
                               )
        error = rs.normal(
                           loc=0.0,
                           scale=error_std,
                           size=size,
                           )
        
        weight = ltruncnorm(
                            loc=weight_mean,
                            scale=weight_std,
                            size=(size, ndim),
                            random_state=rs,
                            )

        climit = ltruncnorm(loc=cnum_mean,
                             scale=cnum_std,
                             size=size,
                             random_state=rs)
        out = (preferences,
               tolerance,
               error,
               weight,
               climit)
        return out
        
    
    def calc_ratings(self, candidates):
        """
        Calculate preference distances & candidate ratings for a given set of candidates
        """
        try:
            candidates = candidates.candidates
        except AttributeError:
            pass
        
        voters = self.voters
        weights = self.weight
        error = self.error
        tol = self.tolerance
        
        if voters.shape[1] == 1:
            weights = None
        rstate = self._randomstate
        
        distances = vcalcs.voter_distances(voters, 
                                           candidates,
                                           weights=weights)
        
        distances = vcalcs.voter_distance_error(distances, 
                                                error,
                                                rstate=rstate)
        
        ratings = vcalcs.voter_scores_by_tolerance(
                                                   None, None,
                                                   distances=distances,
                                                   tol=tol,
                                                   cnum=None,
                                                   strategy='abs',
                                                   )
        self.ratings = ratings
        self.distances = distances
        return ratings
    
    
    def stats(self):
        s = metrics.ElectionStats(voters=self.voters,
                                  weights=self.weights
                                  )
        return s.stats
    


def load_election(fname):
    """Load a pickled Election object from file"""
    with open(fname, 'rb') as f:
        e = pickle.load(f)
    return e





    
def __plot1d(election, results, title=''):
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
    

    
def _plot_hist(output):
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



def __plot2d(election, results, title=''):

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
        
