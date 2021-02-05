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

- `BallotGenerator` takes voter and candidate information to generate honest 
  and tactical ballots.
- `eRunner` handles the running of specific types of elections.
- `ElectionResult` handles the storage of output data. 

"""
import collections
import pickle
import copy
from typing import List, NamedTuple, Tuple, Dict

import numpy as np
import pandas as pd
import scipy
from scipy.stats import truncnorm

from votesim import metrics
from votesim import ballot
from votesim import votemethods
from votesim import utilities
from votesim.models import vcalcs
from votesim.models.dataclasses import (VoterData,
                                        VoterGroupData,
                                        CandidateData, 
                                        ElectionData,
                                        ElectionResult,
                                        strategy_data,
                                        StrategyData,
                                        )
from votesim.strategy import TacticalBallots
# from votesim.strategy import TacticalBallots, FrontRunners

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
    tol : float or None
        Voter preference max tolerance.
    base : str
        Voter rating mapping to distance, either:
            - 'linear' - Linear mapping of distance to rating
            - 'quadratic' - Quadratic mapping of distance to rating
            - 'sqrt' - Square root mappiong of distance to rating
    order : int
        Order or norm calculation for voter-candidate regret distance.
        
    
    Attributes
    ----------
    data : `votesim.models.dataclasses.VoterData`
        Voter data

    """
    
    data: VoterData
    
    def __init__(self, seed: int=None, tol: float=None,
                 base: str='linear', order: int=1):
        self.init(seed, order=order)            
        self.set_behavior(tol=tol, base=base)
        return
        

    @utilities.recorder.record_actions(replace=True)
    def init(self, seed: int, order: int):
        """Set pseudorandom seed & distance calculation order."""
        self.seed = seed
        self._randomstate = _RandomState(seed, VOTERS_BASE_SEED)  
        self._order = order
        self._weights = None
        return self
    
    
    @utilities.recorder.record_actions(replace=True)
    def set_behavior(self, tol: float=None, base: str='linear',):
        """Set voter strategy type."""
        self._tol = tol
        self._base = base

        return self

        
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
            
        return self._add_voters(voters)
    
    
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
        return self
    
    
    @utilities.recorder.record_actions()
    def add(self, pref):
        """Add arbitrary voters.
        
        Parameters
        ----------
        pref : array shape (a, b)
            Voter preferences, `a` is number of voters, `b` pref. dimensions.
        """
        return self._add_voters(pref)
    
    
    
    def _add_voters(self, pref):
        """Base function for adding 2d array of candidates to election."""
        
        try:
            pref = np.row_stack((self._pref, pref))
        except (AttributeError, ValueError):
            pref = np.atleast_2d(pref)
        self._pref = pref            
        return self
    
    
    def build(self):
        """Finalize Voter, construct immutable VoterData."""
        self.data = VoterData(pref=self._pref,
                              weights=self._weights,
                              order=self._order,
                              stats=None,
                              tol=self._tol,
                              base=self._base,
                              )
        return self


    def calculate_distances(self, candidates: CandidateData):
        """Preference distances of candidates from voters for building ballots. 

        Parameters
        ----------
        candidates : votesim.models.dataclasses.CandidateData
            Candidate preference data
        """        
        pref = self.data.pref
        order = self.data.order
        weights = self.data.weights
        
        distances = vcalcs.voter_distances(voters=pref,
                                           candidates=candidates.pref,
                                           weights=weights,
                                           order=order)
        return distances
    
    
    def honest_ballots(self, candidates: CandidateData):
        """Honest ballots calculated from Candidates."""
        distances = self.calculate_distances(candidates)
        b = ballot.gen_honest_ballots(distances=distances,
                                      tol=self.data.strategy['tol'],
                                      base=self.data.strategy['base'])
        return b
    
  
    

class VoterGroup(object):
    """Group together multiple voter objects & interact with candidates.
    
    Parameters
    ----------
    voters_list : list[Voters]
        List of Voters
    
    Attributes
    ----------
    group : list[Voters]
        Same as voters_list
    
    """
    def __init__(self, voters_list: List[Voters]):        
        try:
            iter(voters_list)
        except Exception:
            voters_list = [voters_list]            
        self.group = voters_list  
        self._build()
        return
        
        
    def _build(self):
        """Finalize VoterGroup, build immutable VoterGroupData."""
        group_datas = tuple(v.build() for v in self.group)
        
        orders = np.array([v.data.order for v in self.group])
        if len(orders) > 0:
            order = orders[0]
            if not np.all(orders == orders[0]):
                raise ValueError('Order of voters in group must all be same.') 
        else:
            order = None
        
        # data = self.group[0]
        # data = data.replace(pref=self._get_pref())
        # self.data = data
        
        pref = self._get_pref()     
        stats = metrics.VoterStats(pref=pref,
                                   weights=None,
                                   order=order)      
        
        group_index = dict(enumerate(self.group_indices))
        
        data = VoterGroupData(groups=group_datas,
                              pref=pref,
                              weights=None,
                              order=order,
                              stats=stats,
                              group_index=group_index,
                              )
        self.data = data        
        return self
    
    
    def build(self):
        """This is a dummy build and does nothing. VoterGroup is auto-built."""
        return self
    
    
    def _get_pref(self):
        vlist = [v.data.pref for v in self.group]
        return np.vstack(vlist)    

    
    def __getitem__(self, key):
        return self.group[key]

  
    @utilities.lazy_property
    def group_indices(self):
        """Row indices to obtain child's voters for all children in the voter
        preference and ballot arrays.
        
        Returns
        -------
        slices : list of slice
            Slice which returns the Voter group, indexed by group number. 
        """
        groups = self.group
        lengths = [len(v.data.pref) for v in groups]
        iarr = np.cumsum(lengths)
        iarr = np.append(0, iarr)
        slices = [slice(iarr[i], iarr[i+1]) for i in iarr[:-1]]
        return slices
    

    
def voter_group(vlist) -> VoterGroup:
    """Group together multiple Voters."""
    if hasattr(vlist, 'group'):
        return vlist
    else:
        return VoterGroup(vlist)
        

    
class Candidates(object):
    """Create candidates for spatial model.
    
    Parameters
    -----------
    voters : `Voters` or `VoterGroup`
        Voters to draw population data. 
    seed : int or None
        Seed for random number generation.
    
    Attributes
    ----------
    pref : array shape (a, b)
        Voter preferences, `a` number of candidates, 
        `b` number of preference dimensions    
    """
    data: CandidateData
    
    def __init__(self, voters: Voters, seed: int=None):
        self._method_records = utilities.recorder.RecordActionCache()
        
        if not hasattr(voters, '__len__'):     
            voters = [voters]
            
        self.voters = voter_group(voters)
        self.set_seed(seed)
        return    
    
    
    @utilities.recorder.record_actions()
    def set_seed(self, seed: int):
        """ Set pseudorandom seed """
        self._seed = (seed, CANDIDATES_BASE_SEED)
        self._randomstate = _RandomState(*self._seed)
        return self
    
    
    def _add_candidates(self, candidates: np.ndarray):
        """Base function for adding 2d array of candidates to election"""
        candidates = np.array(candidates)
        assert candidates.ndim == 2, 'candidates array must have ndim=2'
        
        vdata = self.voters.data
        
        try:
            candidates = np.row_stack((self._pref, candidates))
        except (AttributeError, ValueError):
            candidates = np.atleast_2d(candidates)
        
        cdim = candidates.shape[1]
        vdim = vdata.pref.shape[1]
        
        condition = cdim == vdim
        s = ('dim[1] of candidates (%s) '
             'must be same as dim[1] (%s) of self.voters' % (cdim, vdim))
            
        assert condition, s
        self._pref = candidates
        return self
       
    
    def reset(self):
        """Reset candidates for a given Voters.
        Delete candidate preferences and records"""
        try:
            self._method_records.reset()
        except AttributeError:
            pass
        try:
            del self.data
        except AttributeError:
            pass
        return
    
    
    @utilities.recorder.record_actions()
    def add_random(self, cnum: int, sdev=2):
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
        std = self.voters.data.stats.pref_std
        mean = self.voters.data.stats.pref_mean
        ndim = std.shape[0]
        
        candidates = rs.uniform(low = -sdev*std,
                                high = sdev*std,
                                size = (cnum, ndim)) + mean
        return self._add_candidates(candidates)
    
    
    @utilities.recorder.record_actions()
    def add(self, candidates: np.ndarray):
        """Add 2d array of candidates to election, record actions
        
        Parameters
        ----------
        candidates : array shape (a, n)
            Candidate preference coordinates.
            
            - a = number of candidates
            - n = number of preference dimensions
            
        Returns
        -------
        out: Candidates
            `self`
        """
        self._add_candidates(candidates)
        return self
    
    
    def build(self):
        """Construct immutable CandidateData needed for simulation."""
        voters = self.voters
        pref = self._pref
        distances = vcalcs.voter_distances(voters=voters.data.pref,
                                           candidates=pref,
                                           weights=voters.data.weights,
                                           order=voters.data.order,
                                           )          
        stats = metrics.CandidateStats(pref=pref,
                                       distances=distances)
        self.data = CandidateData(pref=self._pref,
                                  distances=distances,
                                  stats=stats)
        return self
            


class StrategiesEmpty(object):
    """Create empty strategy; honest election."""
    _strategies = []
    data = ()  
    
    def __init__(self):
        return
    
    def build(self):
        return self
    
    
    def __len__(self):
        return 0
            

class Strategies(object):
    """Strategy constructor for `VoterGroup`."""
    def __init__(self, vgroup: VoterGroup):
        self._method_records = utilities.recorder.RecordActionCache()
        self.voters = voter_group(vgroup)
        self.vlen = len(self.voters.group)        
        self._strategies = []
        return
    
    @utilities.recorder.record_actions()
    def add(self,
            tactics: tuple,
            subset: str,
            ratio: float,
            underdog: int,
            groupnum: int=0,
            frontrunnertype: str='eliminate',):
        """Set a strategy for a specified voter group.

        Parameters
        ----------
        tactics : tuple[str]
            Tuple of tactic names.
        subset : str
            - '' -- No subset
        - 'topdog' -- Topdog voter coalition
        - 'underdog' -- Underdog voter coalition
        ratio : float
            Ratio of strategic voters in the group from [0 to 1].
        underdog : int or None
            Specify the underdog candidate. Set to None to estimate best underdog.
        groupnum : int, optional
            Voter group number to set strategy. The default is 0.
        frontrunnertype : str, optional
            Strategy used to determine underdog frontrunner.
            The default is 'eliminate'.

        Returns
        -------
        Strategies
            Returns `self`.

        """        
        return self._set(
            tactics=tactics,
            subset=subset, 
            ratio=ratio,
            underdog=underdog,
            groupnum=groupnum,
            frontrunnertype=frontrunnertype,
            )
    
    
    @utilities.recorder.record_actions()
    def fill(self, 
             tactics: tuple,
             subset: str,
             ratio: float,
             underdog: int,
             groupnum: int,
             frontrunnertype='eliminate'):
        """Set strategy for unset groups."""
        locations = self.get_no_strategy
        for ii in locations:
            self._set(
                tactics=tactics,
                subset=subset, 
                ratio=ratio,
                underdog=underdog,
                groupnum=ii,
                frontrunnertype=frontrunnertype,
                )
        return self
           
    
    def _set(self,
             tactics: tuple,
             subset: str,
             ratio: float,
             underdog: int,
             groupnum: int,
             frontrunnertype='eliminate',
             ):
        group_index = self.voters.group_indices[groupnum] 
        strat_data = StrategyData(tactics=tactics,
                         subset=subset,
                         ratio=ratio,
                         underdog=underdog,
                         groupnum=groupnum,
                         index=group_index,
                         frontrunnertype=frontrunnertype)
        self._strategies.append(strat_data)
        return self
    
    
    def build(self):
        """Construct static data needed to run simulation"""
        if len(self.get_no_strategy()) > 0:
            raise ValueError('Insufficient strategies have been defined!')
        if self.has_duplicates():
            raise ValueError('Duplicate strategy entries found.')
        self.data = tuple(self._strategies)
        return self
    
    
    def get_no_strategy(self):
        """ndarray : Groups' index locations that have no strategies set."""
        
        no_strat_locs = []
        for ii, index in enumerate(self.voters.group_indices):
            found = False
            for strategy in self._strategies:
                if np.all(index == strategy.index):
                    found = True
            
            if not found:
                no_strat_locs.append(ii)
        return np.array(no_strat_locs)
    
    
    def has_duplicates(self):
        """Make sure no duplicate group index + subset locations have been defined.
        Return True if duplicates found. False otherwise."""
        data = []
        for strategy in self._strategies:
            index = strategy.index
            subset = strategy.subset
            data.append((repr(index), subset))
        
        count = collections.Counter(data)
        
        count_values = list(count.values())
        iimax = np.argmax(count_values)
        
        if count_values[iimax] > 1:
            logger.warn('Duplicate strategy found at strategy #%s', iimax)
            return True
        return False
    
    
    def __len__(self):
        return len(self._strategies)
            

class BallotGenerator(object):
    """
    Generate ballots from voter and candidate data.
    
    Parameters
    ----------
    voters_list : list of Voter or VoterGroup
        Voters of election
    candidates : Candidates
        Candidates of election
    scoremax : int
        Maximum score for scored ballots.
    """
    tacticalballots : TacticalBallots
    honest_ballot_dict : dict
    
    def __init__(self, 
                 voters_list: VoterGroup,
                 candidates: Candidates,
                 scoremax: int):
        self.candidates = candidates
        self.votergroup = voter_group(voters_list)
        self.scoremax = scoremax
        self._init_honest_builder()
        return
    
    
    def _init_honest_builder(self):
        """Honest ballot constructor for ratings, ranks, scores, and votes."""
        cdata = self.candidates.data
        blist = []
        for voter in self.votergroup.group:
            distances = voter.calculate_distances(cdata)
            b = ballot.gen_honest_ballots(distances=distances,
                                          tol=voter.data.tol,
                                          base=voter.data.base,
                                          maxscore=self.scoremax,)
            blist.append(b)
        self.honest_ballot_gen = ballot.CombineBallots(blist)
        
        bdict = {}
        bdict['rank'] = self.honest_ballot_gen.ranks
        bdict['score'] = self.honest_ballot_gen.scores
        bdict['rate'] = self.honest_ballot_gen.ratings
        bdict['vote'] = self.honest_ballot_gen.votes
        self.honest_ballot_dict = bdict
        return 
        

    def get_honest_ballots(self, etype: str) -> Dict[str, np.ndarray]:
        """Get honest ballot data.

        Parameters
        ----------
        etype : str
            Election method name.

        Returns
        -------
        dict[ndarray]
            Dict with all types of ballots, with keys:
                - 'rank' -- Ranked ballots
                - 'score' -- Scored ballots (int)
                - 'rate' -- Rated ballot data (float)
                - 'vote' -- FPTP votes
        """
        btype = votemethods.get_ballot_type(etype)
        return self.honest_ballot_dict[btype] 
    
    
    def get_ballots(self, etype, strategies=(), result=None, ballots=None):
        """Retrieve tactical ballots.
        
        Parameters
        ----------
        etype : str
            Election method
        strategies : list of `StrategyData`
            Voter strategies to apply onto ballots
        result : `ElectionResult`
            Previous results which can be used to calculate front runner.
            
        Returns
        -------
        ballots : ndarray (v, c)
            New ballots
        group_index : dict
            Index locations of voter groups. 
        
        """
        if len(strategies) == 0:
            ballots = self.get_honest_ballots(etype)
            group_index = self.votergroup.data.group_index
        else:
            if ballots is None:
                ballots = self.get_honest_ballots(etype)
            if result is None:
                raise ValueError('A previous honest result must be provided for tactical ballots.')

            tballot_gen = TacticalBallots(etype=etype,
                                          strategies=strategies,
                                          result=result,
                                          ballots=ballots)
            ballots = tballot_gen.ballots
            group_index = tballot_gen.group_index
            
            # Just save this thing might be useful for debugging. 
            self.tacticalballots = tballot_gen
            
        return ballots, group_index

    

class Election(object):
    """
    Run an Election with Voters and Candidates
    
    Parameters
    ------------
    voters : None, Voters, VoterGroup, or list of Voters
        Voters object specifying the voter preferences and behavior.
    candidate : None or Candidates
        Candidates object specifying candidate preferences
    seed : None
        THIS PARAMETER IS NOT REALLY USED FOR NOW. IGNORE!
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
    ballotgen : BallotGenerator
        VoterBallot data
    
    """
    
    candidates: Candidates
    voters: VoterGroup
    data: ElectionData
    result: ElectionResult
    strategies: Strategies
    ballotgen : BallotGenerator
    

    def __init__(self,
                 voters: VoterGroup=None,
                 candidates: Candidates=None,
                 strategies: Strategies=None,
                 seed=0,
                 numwinners=1,
                 scoremax=5,
                 name = '',
                 save_args=True,
                 save_records=True):
        
        self._method_records = utilities.recorder.RecordActionCache()
        
        self.voters: VoterGroup = None
        self.candidates: Candidates = None
        self.ballotgen: BallotGenerator = None
        self.strategies: Strategies = StrategiesEmpty()
        
        self.save_args = save_args
        self.save_records = save_records
        
        self.init(seed, numwinners, scoremax, name)
        self.set_models(voters, candidates, strategies)
        self._result_calc = ElectionResultCalc(self)
        return
    
    
    @utilities.recorder.record_actions(replace=True)
    def init(self, seed, numwinners, scoremax, name):
        """Initialize some election properties"""
        self._set_seed(seed)
        self.numwinners = numwinners
        self.scoremax = scoremax
        self.name = name
        return
    
    
    def set_models(
            self, 
            voters: Voters=None,
            candidates: Candidates=None,
            strategies: Strategies=None,
            ):
        """Set new voter or candidate model.
        
        Parameters
        ----------
        voters : Voters or None
            New voters object.
            If None, use previously inputed Voters.
        candidates : Candidates or None
            New candidates object. 
            If None, use previously inputed Candidates. 
        strategies : `votesim.models.spatial.Strategies`
            New strategies object.
            If None, use the previously inputed Strategies 
        """
        if voters is not None:
            self.voters = voter_group(voters)

        if candidates is not None:
            
            self.candidates = candidates.build()
            if self.voters is not None:
                self.ballotgen = BallotGenerator(
                    self.voters,
                    self.candidates,
                    scoremax=self.scoremax
                    )
        if strategies is not None:
            if len(strategies) > 0:
                self.strategies = strategies.build()
            else:
                self.strategies = strategies
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
        raise NotImplementedError('This function probably doesnt work.')
        return
    
    
    @utilities.recorder.record_actions(
        replace=True, 
        exclude=['ballots', 'result'])
    def run(self, 
            etype=None,
            ballots=None,
            result=None,
            force_honest=False) -> ElectionResult:
        """Run the election using `votemethods.eRunner`.
        
        Parameters
        ----------
        etype : str
            Election method. Either `etype` or `method` must be input.
        ballots : ndarray
            Initial ballots to be used in election.
        
        result : ElectionResult
            Election, you can input honest election 
            using this object to reduce repetitive computation cost.
        force_honest : bool
            Force run of an honest election without strategy 
        """
        
        return self._run(
            etype=etype,
            ballots=ballots, 
            result=result,
            force_honest=force_honest
            )
        
    
    def _run(self,
            etype=None,
            ballots=None, 
            result=None,
            force_honest=False) -> ElectionResult:

        logger.debug('Running %s, %s, %s', etype)
        strategies = self.strategies.data
        if force_honest:
            strategies = ()
        
        # Auto run an honest election if result is not available.
        elif len(strategies) > 0 and result is None and ballots is None:
            result = self._run(
                etype=etype, 
                ballots=None,
                result=None,
                force_honest=True)
            
        # Retrieve some tactical ballots from honest data. 
        ballots, group_index = self.ballotgen.get_ballots(
            etype=etype,
            strategies=strategies,    
            result=result,
            ballots=ballots
            )
        
        
        # Generate a deterministic seed based on candidates and voters
        runner = votemethods.eRunner(
            etype=etype,
            numwinners=self.numwinners,
            ballots=ballots,
            seed=self._tie_seed(),
            # rstate=self._randomstate,
            )
        self.data = ElectionData(
            ballots=runner.ballots, 
            winners=runner.winners,
            ties=runner.ties,
            group_index=group_index
            )
        
        self.result = self._result_calc.update(
            runner=runner,
            voters=self.voters.data,
            candidates=self.candidates.data,
            election=self.data
            )
        return self.result
    
    
    def _tie_seed(self):
        """Generate pseudorandom seed for tie breaking."""
        v = self.voters.data.pref[0,0] * 1000
        c = self.candidates.data.pref[0,0] * 10000
        return int(abs(v) + abs(c))
  
        
    def rerun(self, d):
        """Re-run an election found in dataframe. Find the election 
        data from the dataframe index.
        
        Parameters
        ----------
        d : dict 
            Dictionary or Series of election data, 
            generated from self.dataseries() or self.dataframe(). 
            
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
        
        filter_key = 'args.strategy.'
        s_dict = filterdict(series, filter_key)
        
        # Construct voters
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
        
        # Construct candidates
        c = type(self.candidates)(voters=new_voters)
        c._method_records.reset()
        c._method_records.run_dict(c_dict, c)
        
        
        # Construct strategies
        s_dict2 = {}
        for k, v in s_dict.items():
            try:
                if not np.isnan(v):        
                    s_dict2[k] = v
            except TypeError:
                s_dict2[k] = v
                
        slen = len(s_dict2)
        if slen > 0:                    
            s = type(self.strategies)(c.voters)
            s._method_records.reset()
            s._method_records.run_dict(s_dict2, s)
        else:
            s = None
        
        enew = Election(voters=c.voters, candidates=c, strategies=s)
        enew._method_records.run_dict(e_dict, enew)
        return enew
    
    
    def copy(self) -> 'Election':
        """Copy election."""
        return copy.copy(self)
       

    def save(self, name, reset=True):
        """Pickle election data.
        
        Parameters
        ----------
        name : str
            Name of new pickle file to dump Election into.
        reset : bool
            If True (default), delete election data that can be regenerated.
        """
        if reset:
            self.reset()
            
        with open(name, 'wb') as file1:
            pickle.dump(self, file1)
        return
    
    
    def dataseries(self, index=None) -> pd.Series:
        """Retrieve pandas data series of output data."""  
        return self._result_calc.dataseries(index=index)
    
    
    def dataframe(self) -> pd.DataFrame:
        """Construct data frame from results history."""
        return self._result_calc.dataframe()
    
    
    def append_stat(self, d: metrics.BaseStats, name='', update_docs=False):
        return self._result_calc.append_stat(d=d,
                                             name=name, 
                                             update_docs=update_docs)
        
        
def calculate_distance(voters: VoterData, candidates: CandidateData):
    """Re-calculate distance as the distance from Election may have error."""
    distances = vcalcs.voter_distances(
                                        voters=voters.pref,
                                        candidates=candidates.pref,
                                        weights=voters.weights,
                                        order=voters.order,
                                        )       
    return distances
    


    
        
class ElectionResultCalc(object):
    """
    Store Election result output. Generated as attribute of Election.
    This is a sort of messy back-end that does all the calculations. The 
    result front end is `ElectionResult`. 
    
    Parameters
    ----------
    e : Election
        Election to extract results from.
        
    Attributes
    ----------
    runner : :class:`~votesim.votemethods.voterunner.eRunner`
        Output from election running class for the last run election. 
    results : dict
        Results of last run election key prefixes:
            
            - 'output.*' -- Prefix for election output results
            - 'args.etype' -- Election method
            - 'args.voter.*' -- Voter input arguments
            - 'args.election.*' -- Election input arguments
            - 'args.user.*' -- User defined input arguments
        
        
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
        # Store results as list of dict
        self._output_history = []
        return
    
    
    def update(self, 
               runner: votemethods.eRunner, 
               voters: VoterData,
               candidates: CandidateData,
               election: ElectionData) -> ElectionResult:
        """Get election results."""
        self.runner = runner   
        self.winners = runner.winners
        self.ties = runner.ties
        self.ballots = runner.ballots
        self.electionStats = metrics.ElectionStats(voters=voters,
                                                   candidates=candidates,
                                                   election=election)
        
        ### Build dictionary of all arguments and output 
        output = {}        
        output.update(self._get_parameters())
        output['output'] = self.electionStats.get_dict()
        output = utilities.misc.flatten_dict(output, sep='.')
        self.output = output      
        
        
        if self.election.save_records:
            self._output_history.append(output)                
        result =  ElectionResult(winners=self.winners,
                                 ties=self.ties,
                                 ballots=self.ballots,
                                 runner=self.runner,
                                 output=self.output,
                                 output_docs=self.output_docs,
                                 stats=self.electionStats,
                                 scoremax=self.election.scoremax
                                 )
        return result
    

        
  
    
    
    def _get_parameter_keys(self) -> list:
        """Retrieve election input parameter keys."""
        return list(self._get_parameters().keys())
    
    
    
    def _get_method_records(self) -> dict:
        """Retrieve records that can be used to regenerate result."""
        candidates = self.election.candidates
        voters = self.election.voters
        strategies = self.election.strategies
        election = self.election

        # get voter parameters
        vrecords = []
        for v in voters.group:
            vrecords.append(v._method_records.dict)
            
        # get candidate parameters
        crecord = candidates._method_records.dict

        # get strategy parameters
        if hasattr(strategies, '_method_records'):
            srecord = strategies._method_records.dict
        else:
            srecord = {}
        
        # get election parameters
        erecord = election._method_records.dict
        
        # Save etype and name in special parameters
        params = {}
        for key in erecord:
            if 'run.etype' in key:
                params['args.etype'] = erecord[key]
            elif '.init.name' in key:
                params['args.name'] = erecord[key]
            
        # Save all method call arguments
        if self.save_args:   
            params['args.candidate'] = crecord
            if len(srecord) > 0:
                params['args.strategy'] = srecord
            for ii, vrecord in enumerate(vrecords):
                params['args.voter-%s' % ii] = vrecord
            params['args.election'] = erecord      
        return params
    
    
    def _get_user_data(self) -> dict:
        # Retrieve user data
        # Determine if user data exists. If not, save default save_args    

        try:
            userdata = self.election._user_data
            if len(userdata)  == 0:
                userdata = {}
        except AttributeError:
            userdata = {}    
            
        params = {}
        # Add user data to params
        for key, value in userdata.items():
            newkey = 'args.user.' + key
            params[newkey] = value    
        return params
        
    def _get_parameters(self) -> dict:
        d1 = self._get_user_data()
        d2 = self._get_method_records()
        d1.update(d2)
        return d1
    
        
    # def ___get_parameters(self) -> dict:
    #     """Retrieve election input parameters."""
    #     params = {}
    #     candidates = self.election.candidates
    #     voters = self.election.voters
    #     election = self.election
        
    #     # get candidate parameters
    #     crecord = candidates._method_records.dict
        
    #     # get voter parameters
    #     vrecords = []
    #     for v in voters.group:
    #         vrecords.append(v._method_records.dict)
        
    #     # get election parameters
    #     erecord = election._method_records.dict
        
    #     # Retrieve user data
    #     # Determine if user data exists. If not, save default save_args    
    #     save_args = self.save_args
    #     try:
    #         userdata = self.election._user_data
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
    
    
    @utilities.lazy_property
    def output_docs(self) -> dict:
        """Retrieve output documentation."""
        docs = self.electionStats.get_docs()
        docs = utilities.misc.flatten_dict(docs, sep='.')
        return docs

    
    def dataseries(self, index=None):
        """Retrieve pandas data series of output data."""  
        if index is None:
            return pd.Series(self.output)
        else:
            return pd.Series(self._output_history[index])
    
    
    def dataframe(self):
        """Construct data frame from results history."""
        
        series = []
        for r in self._output_history:
            series.append(pd.Series(r))
        df = pd.concat(series, axis=1, ignore_index=True).transpose()
        df = df.reset_index(drop=True)
        self._dataframe = df.infer_objects()
        return df

    
    def append_stat(self, d: metrics.BaseStats, name='', update_docs=False):
        """Append custom user stat object to the last result entry.
        
        Parameters
        ----------
        d : subtype of `metrics.BaseStats` or dict
            Additional outputs to add to the result.
        name : str
            Optional, name of outputs.
        """
        try:
            dict1 = d._dict
            docs1 = d._docs
            name1 = d._name
        except AttributeError:
            dict1 = d
            name1 = name
            docs1 = {}    
                
        dict1 = {'output.' + name1 : dict1}
        dict1 = utilities.misc.flatten_dict(dict1, sep='.')
        
        result = self._output_history[-1]
        for key in dict1:
            if key in result:
                s = 'Duplicate output key "%s" found for custom stat.' % key
                raise ValueError(s)
        result.update(dict1)
        return
    


class ResultRecord(object):
    """Store election results here."""
    def __init__(self):
        self.output_history = []
    
        
    def append(self, result: ElectionResult):
        output = result.output
        self.output = output
        self.output_history.append(output)
        return
    
    
    def dataseries(self, index=None):
        """Retrieve pandas data series of output data."""  
        if index is None:
            return pd.Series(self.output)
        else:
            return pd.Series(self.output_history[index])
    
    
    def dataframe(self):
        """Construct data frame from results history."""
        
        series = []
        for r in self.output_history:
            series.append(pd.Series(r))
        df = pd.concat(series, axis=1, ignore_index=True).transpose()
        df = df.reset_index(drop=True)
        self._dataframe = df.infer_objects()
        return df
    
        
    
    def append_stat(self, d: metrics.BaseStats, name='', update_docs=False):
        """Append custom user stat object to the last result entry.
        
        Parameters
        ----------
        d : subtype of `metrics.BaseStats` or dict
            Additional outputs to add to the result.
        name : str
            Optional, name of outputs.
        """
        try:
            dict1 = d._dict
            docs1 = d._docs
            name1 = d._name
        except AttributeError:
            dict1 = d
            name1 = name
            docs1 = {}    
                
        dict1 = {'output.' + name1 : dict1}
        dict1 = utilities.misc.flatten_dict(dict1, sep='.')
        
        result = self.output_history[-1]
        for key in dict1:
            if key in result:
                s = 'Duplicate output key "%s" found for custom stat.' % key
                raise ValueError(s)
        result.update(dict1)
        return
    