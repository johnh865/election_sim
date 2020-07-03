# -*- coding: utf-8 -*-

"""
Generate voter ballots


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

from votesim import votesystems
from votesim import utilities
from votesim.models.vcalcs import distance2rank

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
        return np.round(self.maxscore * self.ratings)
        
        
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
        """Set data of this Ballot object to arguments.
        
        Parameters
        ----------
        ballots : :class:`~votesim.ballots.BallotClass` subclass
            Ballots object
            
        """
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
    
    
    @property
    def votes(self):
        """Plurality votes constructed from ranks."""
        return votesystems.tools.getplurality(ranks=self.ranks)


    def run(self, etype, rstate=None, numwinners=1) -> votesystems.eRunner:
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
        :class:`~votesim.votesystems.eRunner`
            eRunner election output object
        """
        assert etype is not None
        er = votesystems.eRunner(etype=etype, 
                                scores=self.scores,
                                votes=self.votes,
                                ranks=self.ranks,
                                ratings=self.ratings,
                                rstate=rstate,
                                numwinners=numwinners,
                                )
        return er
    
    
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
        """Indices to obtain child's voters for all children.
        
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
        
        Cut rankings where ratings are less than or equal to zero 
        (ie, candidates outside tolerance).
        """    
        ii = self.ratings <= 0
        self.ranks[ii] = 0
        return self
    

class TacticalBallots(BaseBallots):
    """Generate tactical ballots.
    
    Parameters
    ----------
    etype : str
        Name of election system
    ballots : subclass of _BallotClass
        Ballots to tacticalize
    numwinners : int
        Number of front runners to consider
    index : array or None (default)
        Index of tactical voters. All voters are tactical if None.
            
        
    Attributes
    ----------
    from_ballots : BallotClass subtype
        Input ballots
    base_ballots : BaseBallots 
        Base ballots constructed from input ballots
    etype : str
        Election method
    front_runners : array shape (numwinners,)
        Candidates most likely to 
    _iall : int array shape (b,)
        Index of tactical voters
    _ibool : bool array shape (c,)
        Truth array of enabled tactical voters
    """
    
    def __init__(self, 
                 etype: str,
                 ballots: BaseBallots,
                 numwinners=2,
                 index=None,
                 onesided=False):
        
        
        self.from_ballots(ballots)
        self.base_ballots = BaseBallots(ballots=ballots)

        self.etype = etype        
        self.onesided = onesided
        self.front_runners = frontrunners(etype, 
                                          ballots=ballots,
                                          numwinners=numwinners)
        self._set_index(index)
        return
    
    
    def _set_index(self, index):
        """Set enabled tactical voter index.
        
        Parameters
        ----------
        index : (a,) array
            Index locations of enabled tactical voters
        
        """
        bnum = len(self.base_ballots.ratings)
        if index is None:
            self.index = slice(None)
            self._iall = np.arange(bnum)
            self._ibool = np.ones(bnum, dtype=bool)
        else:
            self.index = index
            self._iall = np.array(index)
            self._ibool = np.zeros(bnum, dtype=bool)
            self._ibool[index] = True
            
        # If one-sided ballots get new index from underdog voters
        if self.onesided:
            new = self._index_underdog
            self._set_index(new)
            
            
  
            
        
    @utilities.lazy_property
    def _best_worst_frontrunner(self):
        frunners = self.front_runners
        distances = self.distances
        fdistances = distances[:, frunners]
        
        c_best = np.argmin(fdistances, axis=1)
        c_best = frunners[c_best]
        c_worst = np.argmax(fdistances, axis=1)
        c_worst = frunners[c_worst]        
        return c_best, c_worst
    
    
    @property
    def preferred_frontrunner(self):
        """Array shape (a,) : Preferred front runner for each voter."""
        return self._best_worst_frontrunner[0]
        
    
    @property    
    def hated_frontrunner(self):
        """Array shape (a,) : Hated front runner for each voter."""
        return self._best_worst_frontrunner[1]
    
    
    @utilities.lazy_property
    def most_hated(self):
        """Most hated candidate in the race."""
        distances = np.sum(self.distances, axis=0)
        return np.argmax(distances)
    
    
    @property
    def projected_winner(self):
        """Projected winner from honest votes."""
        return self.front_runners[0]
    
    @utilities.lazy_property
    def _index_under_top(self):
        index = self.index
        
        bnum = len(self.base_ballots.ratings)
        ibool = np.zeros(bnum, dtype=bool)
        ibool[index] = True
        
        winner = self.projected_winner
        itop = self.preferred_frontrunner == winner
        ibot = ~itop
        
        iitop = itop & ibool        
        index_topdog  = np.where(iitop)[0]
        
        iibot = ibot & ibool
        index_underdog = np.where(iibot)[0]
        return index_topdog, index_underdog


    @property
    def index_underdog(self):
        """int array: Index locations of underdog voters who shall tactically vote."""
        return self._index_under_top[0]
        
    
    @property
    def index_topdog(self):
        """int array: Index locations of topdog voters who shall honestly vote."""
        return self._index_under_top[1]
            
    
    def compromise(self):
        """Maximize preference in favor of favorite front runner."""
        b = self
        b.ratings[self._iall, self.preferred_frontrunner] = 1
        
        ranks1 = b.ranks.astype(float)
        ranks1[self._iall, self.preferred_frontrunner] = 0.5
        ranks1 = votesystems.tools.rcv_reorder(ranks1)
        b.ranks = ranks1
        return b
    
    
    def bury(self):
        """Bury hated front-runner."""
        b = self
        b.ratings[self._iall, self.hated_frontrunner] = 0
        b.ranks[self._iall, self.hated_frontrunner] = 0        
        return b
        
    
    def truncate_hated(self):
        """Truncate all candidates equal or worse than hated front-runner."""
        dist_hated = self.distances[:, self.hated_frontrunner]
        idelete  = self.distances >= dist_hated
        idelete = idelete & self._ibool
        
        b = self
        b.ratings[idelete] = 0
        b.ranks[idelete] = 0
        return b
    
    
    def truncate_preferred(self):
        """Truncate all candidates worst than favorite frontrunner."""
        dist_fav = self.distances[:, self.preferred_frontrunner]
        idelete = self.distances > dist_fav
        idelete = idelete & self._ibool
        b = self
        b.ratings[idelete] = 0
        b.ranks[idelete] = 0
        return b
        
        
    def bullet_preferred(self):
        """Bullet vote for preferred front runner."""
        b = self
        b.ratings[self._iall, :] = 0
        b.ratings[self._iall, self.preferred_frontrunner] = 1
        
        b.ranks[self._iall, :] = 0
        b.ranks[self._iall, self.preferred_frontrunner] = 1
        return b
    
    
    def bullet_favorite(self):
        """Bullet vote for your favorite candidate."""
        b = self
        
        favorites = np.argmin(self.distances, axis=1)
        b.ratings[self._iall, :] = 0
        b.ratings[self._iall, favorites] = 1
        b.ranks[self._iall, :] = 0
        b.ranks[self._iall, favorites] = 1
        return b
    
    
    def minmax_hated(self):
        """Max-min hated strategy.
        
        Zeroing all candidates equal or worse than 
        hated front-runner.
        """
        b = self.truncate_hated()
        ij_bool = (b.ratings > 0) * self._ibool[:, None]
        b.ratings[ij_bool] = 1.0
        return b
    
    
    def minmax_preferred(self):
        """Max-min favorite strategy.
        
        Zero all candidates equal or worse than 
        favorite front-runner.
        """
        b = self.truncate_preferred()
        ij_bool = (b.ratings > 0) * self._ibool[:, None]
        b.ratings[ij_bool] = 1.0
        return b    
    
    
    def set(self, tactics=(), onesided=False, index=None):
        """
        Parameters
        ----------
        tactics : list of str
            List of tactics to be applied. Possible values are
            
            - 'compromise'
            - 'bury'
            - 'truncate_hated'
            - 'truncate_preferred'
            - 'bullet_preferred'
            - 'bullet_favorite'
            - 'minmax_hated'
            - 'minmax_preferred'
        onesided : bool, optional
            Use one-sided strategy. The default is False
        index : int array, bool array, None, optional
            Voter index locations to apply strategy. The default is None.
        """
        self.onesided = onesided
        self._set_index(index)
        for name in tactics:
            getattr(self, name)()
            
            

def frontrunners(etype: str, ballots: BaseBallots, numwinners=2,):
    """Get front runners of election given Ballots.

    Parameters
    ----------
    etype : str
        Election etype name.
    numwinners : int, optional
        Number of front-runners to retrieve. The default is 2.

    Raises
    ------
    ValueError
        Thrown if no winners or ties output from election method.

    Returns
    -------
    array shape (numwinners+, )
        Frontrunners of the election. The number returned may be greater than
        specified if ties are found. The first entry will be the projected
        winner. 
    """
    frunners = []
    ballots = ballots.copy()
    er = ballots.run(etype, numwinners=1)
    
    try:
        # Some voting systems have an output called 'talley' which returns
        # tallys for each candidate that determine the winner. 
        # For these systems use tally to get the frontrunners. 
        tally = er.output['tally']
        winners, ties = votesystems.tools.winner_check(tally,
                                                       numwin=numwinners)
        return np.append(winners, ties)
    
    except (KeyError, TypeError):
        pass
    
    # If tally doesn't exist, determine front runner by rerunning
    # the election without the winner.
    while len(frunners) < numwinners:

        winners = er.winners
        ties = er.ties

        if len(winners) > 0:
            frunners.extend(winners)
            ballots.ratings[:, winners] = 0
            
        elif len(ties) > 0:
            frunners.extend(ties)
            ballots.ratings[:, winners] = 0
        else:
            raise ValueError('No ties or winners found for election method.')
            
        er = ballots.run(etype, numwinners=1)
            
    return np.array(frunners)
                

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
    

# def gen_tactical_ballots(ballots: BallotClass,
#                          etype: str,
#                          tactics=(),
#                          onesided=False, index=None) ->TacticalBallots:    
#     """
#     etype : str
#         Election Type.    
#     tactics : list or str, optional
#         Tactical manipulations to apply onto ballot. The default is ().
#     onesided : bool, optional
#         Return one-sided ballots, where only projected losers strategically
#         vote. The default is False.
#     """
#     if len(tactics) == 0:
#         return ballots
    
#     ballots = TacticalBallots(etype, ballots, index=index)
    
#     if  onesided:
#         ballots = OneSidedBallots(etype, ballots, index=index)
#     else:
#         ballots = TacticalBallots(etype, ballots, index=index)        
#     for name in tactics:
#         ballots = getattr(ballots, name)()
        
#     return ballots


                     