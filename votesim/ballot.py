# -*- coding: utf-8 -*-

"""
Generate voter ballots.


Ballot generation parameters 
-----------------------------
norm : bool
    Normalize ratings
cut_ranks : bool
    Cut away ranks according to voter tolerance
compromise : int (0 default)
    Number of frontrunners to consider in compromise strategy.
    Compromise according to the top specified candidates. 
    Set to zero for no strategy
bury : int (0 default)
    Number of frontrunners to consider in bury strategy.
    Bury according to the top specified candidates. 
    Set to zero for no strategy


"""
import numpy as np
import copy

import votesim
from votesim import votemethods
from votesim import utilities
from votesim.models.vcalcs import distance2rank
from votesim.metrics.metrics import regret_tally

__all__ = ['gen_honest_ballots',
           'TacticalBallots',
           'CombineBallots',
           'BaseBallots',
           'BallotClass',
           ]

class BallotClass(object):
    """
    Base BallotClass class used to create Ballot sub-classes.
    
    Parameters
    ----------
    ranks : array shape (a, b) or None (default)
        Input rank data for `a` voters and `b` candidates, where
        1 is the most preferred rank and 0 is unranked.
    ratings : array shape (a, b) or None (default)
        Input ratings data for `a` voters and `b` candidates from 0 to 1.
    distances : array shape (a, b) or None (default)
        Voter regret, or preference distance away from each candidate
    tol : (a,) array, float, or None
        Voter preference tolerance at which candidate ratings are less than zero. 
        The default is None.
    rtol : (a,) array, float, or None
        Relative voter preference tolerance based on worst candidate.
        Either tol or rtol can be specified, but not both.
        The default is None.        
        
        
    udata : array shape (a, c)
        User input data that may be used in Ballot subclasses
    maxscore : int (default 5)
        Maximum integer score for score ballot generation
    ballots : :class:`~votesim.ballots.BallotClass` subclass
        Ballot to read information from.        
    """
    
    def __init__(self, ranks=None, ratings=None, distances=None, tol=None,
                 rtol=None,
                 udata=None,
                 maxscore=5, ballots=None,):
        
        
        
        if ballots is not None:
            self.from_ballots(ballots)
        else:
            s = 'if ballots is None, distances & tol must be defined'
            assert distances is not None, s
            #assert tol is not None, s
            
            self.ranks = ranks
            self.ratings = ratings
            self.distances = distances
            
            tol = self._init_tol(tol, rtol)
            self.tol = tol
            self.maxscore=maxscore
            self.udata = udata
            
        self._init_subclass()
        
        
    @utilities.lazy_property
    def scores(self):
        """Generate integer scores from ratings."""
        if self.ratings is None:
            raise ValueError('Ratings must be generated to retrieve scores.')
        return np.round(self.maxscore * self.ratings)
        
    
    @utilities.lazy_property
    def votes(self):
        """Plurality votes constructed from ranks."""
        if self.ranks is None:
            raise ValueError('Ranks must be generated to retrieve votes.')
        return votemethods.tools.getplurality(ranks=self.ranks)
        
        
    def _init_subclass(self):
        """You can stick in custom initialization routines for subclasses here."""
        return
    
    
    def _init_tol(self, tol, rtol):
        """Manipulate tol so that it is an acceptable parameter.
        Handle default values; rtol is for relative tolerance."""
        if tol is None and rtol is None:
            tol = self.relative_tol
                             
        elif rtol is not None:
            tol = self.relative_tol * rtol
            
        else:
            tol = np.array(tol)

        if rtol is not None and tol is not None:
            raise ValueError('Only either rtol OR tol can be specified.')
            
            
        if tol.ndim == 1:
            tol = tol[:, None]
        return tol    
        
    
    def from_ballots(self, ballots):
        """Set data of an Ballot object to arguments.
        
        Parameters
        ----------
        ballots : :class:`~votesim.ballots.BallotClass` subclass
            Ballots object
            
        """
        ballots = ballots.copy()
        self.ranks = ballots.ranks
        self.ratings = ballots.ratings
        self.distances = ballots.distances
        self.tol = ballots.tol
        self.maxscore = ballots.maxscore            
        self.udata = ballots.udata
        return
    
    
    # @property
    # def ranks(self):
    #     if self._ranks is None:
    #         raise AttributeError('ballot ranks not yet defined.')
    #     return self._ranks
    
    
    # @property
    # def ratings(self):
    #     if self._ratings is None:
    #         raise AttributeError('ballot ratings not yet defined.')        
    #     return self._ratings
    
    
    # @property
    # def distances(self):
    #     if self._distances is None:
    #         raise AttributeError('ballot distances not yet defined.')
    #     return self._distances
    
    
    # @property
    # def tol(self):
    #     if self._tol is None:
    #         raise AttributeError('ballot tol not yet defined.')        
    #     return self._tol
        
    
    
    def copy(self):
        """Return copy of Ballots."""
        return copy.deepcopy(self)
    


    def run(self, etype, rstate=None, numwinners=1) -> votemethods.eRunner:
        """Run election method on ballots.
        
        Parameters
        ----------
        etype : str
            Name of election method 
        rstate : numpy.random.RandomState
            Random number generator
        numwinners : int
            Number of winners desired.
        
        Returns
        -------
        :class:`~votesim.votemethods.eRunner`
            eRunner election output object
        """
        assert etype is not None
        ballots = self.get_ballots(etype)
        er = votemethods.eRunner(etype=etype,
                                 ballots=ballots,
                                 rstate=rstate,
                                 numwinners=numwinners,)
        self._erunner = er
        return er
    
    
    def get_ballots(self, etype: str):
        """Retrieve the ballots needed for an election method."""
        if etype in votemethods.ranked_methods:
            ballots = self.ranks
        elif etype in votemethods.vote_methods:
            ballots = self.votes
        elif etype in votemethods.scored_methods:
            ballots = self.scores
        elif etype in votemethods.rated_methods:
            ballots = self.ratings
        return ballots
    
    
    def set_erunner(self, erunner):
        if erunner is not None:
            self._erunner = erunner
        return
    
    
    @property
    def erunner(self) -> votemethods.eRunner:
        """eRunner object from last run election."""
        try:
            return getattr(self, '_erunner')
        except AttributeError:
            raise AttributeError('self.run(...) must first be executed to access erunner.')
                
    
    def chain(self, s):
        """Chain together multiple ballot manipulation functions, call
        by the method name
        
        
        Examples
        ------------
        
        >>> s = 'rank_honest.rate_linear.compromise.bury'
        >>> out = self.chain(s)
        
        """
        
        cmds = s.split('.')
        
        obj = self
        for cmd in cmds:
            obj = getattr(obj, cmd)()
        return obj
    
    
    def set_maxscore(self, maxscore):
        """Set the maximum score for scored ballots."""
        b = self.copy()
        b.maxscore = maxscore
        return b
    
    
    @utilities.lazy_property
    def relative_tol(self):
        """Set tolerance relative to the worst candidate."""
        dmax = np.max(self.distances, axis=1)
        return dmax
            

class CombineBallots(BallotClass):
    """Combine multiple ballot objects.
    Not everything is combined, only ranks, ratings, and distances.
    
    Parameters
    ----------
    children : list of type :class:`~votesim.ballots.BallotClass`
        Ballots to combine.
    """
    
    def __init__(self, children):
        list_ranks = [b.ranks for b in children]
        list_ratings = [b.ratings for b in children]
        list_dist = [b.distances for b in children]
        
        maxscore = children[0].maxscore
        ranks = np.vstack(list_ranks)
        ratings = np.vstack(list_ratings)
        distances = np.vstack(list_dist)
        
        self.ranks = ranks
        self.ratings = ratings
        self.distances = distances
        self.tol = None
        self.maxscore = maxscore
        self.udata = None
        self.children = children
        self._init_subclass()        
        
    
    @utilities.lazy_property
    def children_indices(self):
        """Row indices to obtain child's voters for all children in the voter
        preference and ballot arrays.
        
        Returns
        -------
        slices : list of slice
            Slice which returns the child
        """
        lengths = [len(child.ranks) for child in self.children]
        iarr = np.cumsum(lengths)
        iarr = np.append(0, iarr)
        slices = [slice(iarr[i], iarr[i+1]) for i in iarr[:-1]]
        return slices
    
        
            
            
        
        
class BaseBallots(BallotClass):
    """Base ballot construction class.
    
    Create variants of honest ballots -- such as setting cutoffs for ranking,
    scaling and setting rating tolerance, etc.
    
    See :class:`~votesim.ballots.BallotClass`.
    """
    
    def rank_honest(self):
        """Construct honest ballots."""
        self.ranks = distance2rank(self.distances)
        return self
    


    def rate_linear(self):
        """Construct ratings as r = (1 - d/tol)."""
        r = (1.0 - self.distances / self.tol)
        r = np.maximum(r, 0)
        self.ratings = r
        return self
        
    
    def rate_quadratic(self):
        """Construct ratings as  r = (1 - d/tol)^2."""
        r = (1.0 - self.distances / self.tol)**2
        r = np.maximum(r, 0)
        self.ratings = r
        return self
    
    
    def rate_sqrt(self):
        """Construct ratings as  r = sqrt(1 - d/tol)."""
        r = (1.0 - self.distances / self.tol)**0.5
        r = np.maximum(r, 0)
        self.ratings = r        
        return self
    
    
    def rate_norm(self):
        
        """Construct normalized ballots; rating of best candidate set to maximum rating."""
        ratings = self.ratings
        max_ratings = np.max(ratings, axis=1)[:, None] 
        i2 = np.where(max_ratings == 0)
        max_ratings[i2] = 1.  # avoid divide by zero error for unfilled ballots        
        factor = 1.0 / max_ratings
        
        self.ratings = ratings * factor        
        return self
    
    
    def rank_cut(self):
        """Cut off rankings of candidates.
        
        Cut rankings where ratings are less than zero 
        (ie, candidates outside tolerance).
        """    
        err_tol = 1e-5
        ii = self.ratings < 0 - err_tol
        self.ranks[ii] = 0
        return self
    

                
def gen_honest_ballots(distances, tol=None, rtol=None, maxscore=5, 
                       base='linear',):
    """
    Create voter ballots.

    Parameters
    ----------
    distances : (a, b) array
        `a` Voter distances from `b` candidates
    tol : (a,) array, float, or None
        Voter preference tolerance at which candidate ratings are less than zero. 
        The default is None.
    rtol : (a,) array, float, or None
        Relative voter preference tolerance based on worst candidate.
        Either tol or rtol can be specified, but not both.
        The default is None.  
    maxscore : int, optional
        Max ballot integer score. The default is 5.
    base : str, optional
        Base ballot type. The default is 'linear'.
        
        - 'linear' - Linear mapping of distance to rating
        - 'quadratic' - Quadratic mapping of distance to rating
        - 'sqrt' - Square root mappiong of distance to rating
        

    Returns
    -------
    ballots : subclass of :class:`~votesim.ballots.BallotClass`
        Constructed voter ballots.

    """    
    ballots = BaseBallots(distances=distances,
                          tol=tol,
                          rtol=rtol, 
                          maxscore=maxscore)
    # names = tactics.split(',')
    # names = [names.strip() for n in names]
    
    if base == 'linear':
        ballots = (ballots.rank_honest()
                  .rate_linear()
                  .rate_norm()
                  .rank_cut()
                  )
    elif base == 'quadratic':
        ballots = (ballots.rank_honest()
                  .rate_quadratic()
                  .rate_norm()
                  .rank_cut()
                  )
    elif base == 'sqrt':
        ballots = (ballots.rank_honet()
                   .rate_sqrt()
                   .rate_norm()
                   .rank_cut()
                   )
    return ballots           