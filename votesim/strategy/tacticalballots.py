# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 18:23:27 2020

@author: John
"""
import numpy as np
import functools

import votesim
from votesim import votemethods
from votesim import utilities
from votesim.models.vcalcs import distance2rank
from votesim.metrics.metrics import regret_tally
from votesim.ballot import BaseBallots
from votesim.utilities.decorators import lazy_property

class TacticalBallots(BaseBallots):
    """Generate tactical ballots.
    
    Parameters
    ----------
    etype : str
        Name of election system
    ballots : subclass of _BallotClass
        Ballots to tacticalize
        
    numwinners : int
        Optional, Number of front runners to consider
    index : array or None (default)
        Optional, Index of tactical voters. All voters are tactical if None.
    onsided : bool
        Optional, If True, only underdog voters vote tactically. 
        Top-dog voters vote honestly. 
    frontrunnertype : str
        Optional, front runner prediction calculation type
        
        - "tally" -- (Default) Attempt to use method specifc tally. 
          Use "elimination" if none found. 
        - "elimination" -- Find runner up by eliminating honest winner.
        - "score" -- Use scored tally
        - "plurality" -- Use plurality tally
    front_runners : list of int, array(f,), or None
        Optional, Top candidates of the election if known. Set to None to 
        calculate front runners. 
    erunner : :class:`~votesim.votemethods.eRunner` or None
        Optional, If an election run is available, you can input it here 
        to reduce computation cost.         
       
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
        Index of all tactical voters
    _ibool : bool array shape (c,)
        Truth array of enabled tactical voters
    """
    
    def __init__(self, 
                 etype: str,
                 ballots: BaseBallots=None,
                 election: 'votesim.spatial.Election'=None,
                 ):
        
        self.from_ballots(ballots)
        self.base_ballots = BaseBallots(ballots=ballots)
        self.etype = etype        
        
        # Store front runner results for all types
        # self.front_runners = front_runners
        # self._set_index(index, name)
        self.set_data(ballots=ballots, election=election)
        return
    

    def set(self, 
            tactics=(), 
            subset='',
            frontrunnertype='tally',
            frontrunnertol=0.0,
            frontrunnernum=2,
            index=None,    
            ):
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
        # Get front runners
        try:
            _old_frunners = self.front_runners
        except AttributeError:
            _old_frunners = []
        self.front_runners = self._get_frontrunner.get_frontrunners(
                                                kind=frontrunnertype,
                                                num=frontrunnernum,
                                                tol=frontrunnertol,
                                                )
        
        # Reset some properties if new front runners.
        if np.array_equal(_old_frunners, self.front_runners):
            pass
        else:
            utilities.clean_some_lazy_properties(self, 
                                                 ['_best_worst_frontrunner',
                                                  '_index_under_top',
                                                  ])
            
        # Set tractical voter index
        self._set_index(index=index, subset=subset)
            
        # Chain through all tactics and call them.
        if isinstance(tactics, str):
            getattr(self, tactics)()
        else:
            for name in tactics:
                getattr(self, name)()
        return
    
    
    def set_data(self, ballots=None, election=None):
        """Set election data"""
        frunners = FrontRunners(etype=self.etype, 
                                election=election,
                                ballots=ballots,)
        self._get_frontrunner = frunners
        
        self.from_ballots(ballots)
        self.base_ballots = BaseBallots(ballots=ballots)
        utilities.clean_lazy_properties(self)
        return
    
                
    # @property
    # def front_runners(self):
    #     """array(f,) : Front runners, retrieved from either  user input;
    #     if no user input found, calculate the front runner by running the 
    #     election using self.base_ballots."""
    #     if self._front_runners is not None:
    #         return self._front_runners
        
    #     etype = self.etype
    #     ballots = self.base_ballots
    #     numwinners = self.numwinners
    #     frontrunnertype = self.frontrunnertype
        
    #     try:
    #         erunner = self.erunner
    #     except AttributeError:
    #         erunner = None
            
    #     new =  frontrunners(etype=etype,
    #                         ballots=ballots,
    #                         numwinners=numwinners,
    #                         kind=frontrunnertype,
    #                         erunner=erunner)
        
    #     self._front_runners = new
    #     return self._front_runners
                
                
    def _set_index(self, index, subset=''):
        """Set enabled tactical voter index.
        
        Parameters
        ----------
        index : (a,) array
            Index locations of enabled tactical voters
        name : str
            Tactical subset, either
                - '' = Use all of index
                - 'underdog = Only set underdog voters
                - 'topdog' = Only set topdog voters
                
        """      
        bnum = len(self.base_ballots.ratings)
        self._iloc_int_all = np.arange(bnum, dtype=int)
        
        if index is None:
            self.index = slice(None)
            #self._iloc_int = np.arange(bnum, dtype=int)
            self._iloc_bool = np.ones(bnum, dtype=bool)
        else:
            self.index = index
            self._iloc_bool = np.zeros(bnum, dtype=bool)
            self._iloc_bool[index] = True
            #self._iloc_int = np.where(self._iloc_bool)[0]
                        
        if subset == 'underdog':
            self._iloc_bool = self.iloc_bool_underdog & self._iloc_bool
        elif subset == 'topdog':
            self._iloc_bool = self.iloc_bool_topdog & self._iloc_bool
        elif subset == '':
            pass
        else:
            raise ValueError(f'subset "{subset}" is not a valid value.')

        self._iloc_int = np.where(self._iloc_bool)[0]
        return
  
        
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
    def hated_candidate(self):
        """Array shape (a,) : Most hated candidate for each voter."""
        return np.argmax(self.distances, axis=1)

    
    
    @property
    def projected_winner(self):
        """Projected winner from honest votes."""
        return self.front_runners[0]
    
    
    @utilities.lazy_property
    def _index_under_top(self):
        """Calculate top dog and under dog index"""
        index = self.index
        
        bnum = len(self.base_ballots.ratings)
        ibool = np.zeros(bnum, dtype=bool)
        ibool[index] = True
        
        winner = self.projected_winner
        itop = self.preferred_frontrunner == winner
        ibot = ~itop
        
        iitop = itop & ibool        
        iloc_bool_topdog = iitop
        #index_topdog  = np.where(iitop)[0]
        
        iibot = ibot & ibool
        iloc_bool_underdog = iibot
        #index_underdog = np.where(iibot)[0]
        return iloc_bool_underdog, iloc_bool_topdog


    @property
    def iloc_bool_underdog(self):
        """int array: Index locations of underdog voters who shall tactically vote."""
        return self._index_under_top[0]
        
    
    @property
    def iloc_bool_topdog(self):
        """int array: Index locations of topdog voters who shall honestly vote."""
        return self._index_under_top[1]
            
    
    def compromise(self):
        """Maximize preference in favor of favorite front runner."""
        b = self
        ii = self._iloc_int
        jj = self.preferred_frontrunner[self._iloc_int]
        b.ratings[ii, jj] = 1
        
        ranks1 = b.ranks.astype(float)
        ranks1[ii, jj] = 0.5
        ranks1 = votemethods.tools.rcv_reorder(ranks1)
        b.ranks = ranks1
        return b
    
    
    def deep_bury(self):
        """Bury hated front-runner even lower than most hated candidate"""
        b = self
        ii = self._iloc_int
        jj = self.hated_frontrunner[self._iloc_int]
        b.ratings[ii, jj] = 0
        b.ranks[ii, jj] = 0        
        return b
    
    
    def bury(self):
        """Bury hated front-runner equal to most hated candidate"""
        b = self
        ii = self._iloc_int
        jj = self.hated_candidate[ii]
        b.ratings[ii, jj] = 0
        b.ranks[ii, jj] = 0
        return self.deep_bury()
        
        
    def truncate_hated(self):
        """Truncate all candidates equal or worse than hated front-runner."""
        
        iall = self._iloc_int_all
        dist_hated = self.distances[iall, self.hated_frontrunner]
        idelete  = self.distances >= dist_hated[:, None]
        idelete = idelete & self._iloc_bool[:, None]
        
        b = self
        b.ratings[idelete] = 0
        b.ranks[idelete] = 0
        return b
    
    
    def truncate_preferred(self):
        """Truncate all candidates worse than favorite frontrunner."""
        
        iall = self._iloc_int_all
        dist_fav = self.distances[iall, self.preferred_frontrunner]
        idelete = self.distances > dist_fav[:, None]
        idelete = idelete & self._iloc_bool[:, None]

        b = self
        b.ratings[idelete] = 0
        b.ranks[idelete] = 0
        return b
        
        
    def bullet_preferred(self):
        """Bullet vote for preferred front runner."""
        b = self
        ii = self._iloc_int
        jj = self.preferred_frontrunner[self._iloc_int]
        
        b.ratings[ii, :] = 0
        b.ratings[ii, jj] = 1
        b.ranks[ii, :] = 0
        b.ranks[ii, jj] = 1
        return b
    
    
    def bullet_favorite(self):
        """Bullet vote for your favorite candidate."""
        b = self
        
        favorites = np.argmin(self.distances, axis=1)
        ii = self._iloc_int
        jj = favorites[ii]
        
        b.ratings[ii, :] = 0
        b.ratings[ii, jj] = 1
        b.ranks[ii, :] = 0
        b.ranks[ii, jj] = 1
        return b
    
    
    def minmax_hated(self):
        """Max-min hated strategy.
        
        Zeroing all candidates equal or worse than 
        hated front-runner.
        """
        b = self.truncate_hated()
        ibool = self._iloc_bool[:, None]
        ij_bool = (b.ratings > 0) * ibool
        b.ratings[ij_bool] = 1.0
        return b
    
    
    def minmax_preferred(self):
        """Max-min favorite strategy.
        
        Zero all candidates equal or worse than 
        favorite front-runner.
        """
        b = self.truncate_preferred()
        ibool = self._iloc_bool[:, None]
        ij_bool = (b.ratings > 0) * ibool
        b.ratings[ij_bool] = 1.0
        return b    
    
    
                
class FrontRunners(object):
    def __init__(self, 
                 etype: str,
                 election: "votesim.spatial.Election"=None,
                 ballots: BaseBallots=None,
                 # kind: str='tally',
                 # num: int=2,
                 # tol: float=0.0,
                 ):
        # self.kind = kind
        # self.num = num
        self.etype = etype
        # self.tol = tol
        
        
        if ballots is not None:
            self.ballots = ballots
            self.runner = ballots.run(etype, numwinners=1)
        else:
            self.ballots = election.ballots
            
        if election is not None:
            self.election = election
            self.runner = self.election.result.runner
        self.winners = self.runner.winners
        return
    
    
    @functools.lru_cache
    def get_frontrunners(self, kind: str, num: int, tol: float):
        """
        Parameters
        ----------
        kind : str
            Either 
                - 'tally'
                - 'eliminate'
                - 'condorcet'
                - 'regret'
        num : int
            Minimum number of candidates to consider.
        tol : float
            Additional candidates may be considered if their tally is close 
            to the honest winner's tally. `tol` controls the 
            ratio from [0 to 1]. 
            If 0, no additional candidates will be considered. 
            
        Returns
        -------
        out : ndarray (a,)
            Election frontrunners with the honest winner as first entry."""
        
        if kind == 'eliminate':
            return self.get_frontrunners_elimination(num)
        else:
            return self.get_frontunners_tally(kind, num, tol)
        
        
        
    @functools.lru_cache
    def get_frontunners_tally(self, kind, num, tol):
        
        # Some voting systems have an output called 'talley' which returns
        # tallys for each candidate that determine the winner. 
        # For these systems use tally to get the frontrunners. 
   
        tally = self.get_tally(kind)
        numwinners = num
        winners, ties = votemethods.tools.winner_check(tally,
                                                       numwin=numwinners)
        frontrunners = list(winners) +  list(ties)
        
        # Check to make sure the winners are in the front runner list
        for winner in self.winners:
            if winner not in frontrunners:
                frontrunners.insert(0, winner)
                            
        # Add the additional "close" frontrunners
        flist2 = additional_tally_frontrunners(tally, tol=tol)
        
        for f in flist2:
            if f not in frontrunners:
                frontrunners.append(f)
                
        return np.array(frontrunners)
    
    
    
    def get_frontrunners_elimination(self, num):
        
        ballots = self.ballots
        er = self.runner
        numwinners = num
        frunners = []
        
        while len(frunners) < numwinners:
            winners = er.winners
            ties = er.ties         
            
            if len(ties) > 0:
                frunners.extend(ties)
                ballots.ratings[:, ties] = 0
                ballots.ranks[:, ties] = 0
                ballots.votes[:, ties] = 0     
                
            elif len(winners) > 0:
                frunners.extend(winners)
                ballots.ratings[:, winners] = 0
                ballots.ranks[:, winners] = 0
                ballots.votes[:, winners] = 0
         
            else:
                raise ValueError('No ties or winners found for election method.')
                
            er = ballots.run(self.etype, numwinners=1)
                    
        return np.array(frunners)
    
    
    def get_tally(self, kind):
        """Metric of how much candidates are winning election which might
        be output by election method."""
        
        if kind == 'tally':
            tally = self.get_tally_method()
            
        elif kind == 'condorcet':
            tally = self.get_tally_condorcet()
            
        elif kind == 'regret':
            tally = self.get_tally_regret()     
        return tally
                
    
    def get_tally_method(self):
        """Retrieve tally generated by election method."""
        try:
            tally = self.runner.output['tally']
            return tally
        
        except (KeyError, TypeError):
            s = f'"tally" could not be find in voting method {self.etype} output.'
            raise TallyError(s)
                
    
    def get_tally_regret(self):
        """Retrieve tally calculated by voter regret."""
        estats = self.election.electionStats
        tally = regret_tally(estats)
        return tally
    
    
    def get_tally_condorcet(self):
        """Retrieve tally from a Condorcet-like process."""
        output = self.runner.output
        margins = output['margin_matrix']
        tally = np.min(margins, axis=0)
        return tally


class TallyError(Exception):
    """Need a special error that can be caught by FrontRunners"""
    pass




    


def frontrunners(etype: str, 
                 ballots: BaseBallots,
                 numwinners: int=2,
                 kind: str='tally',
                 erunner: votemethods.eRunner=None,
                 election: "votesim.spatial.Election"=None):
    """Get front runners of election given Ballots.

    Parameters
    ----------
    etype : str
        Election etype name.
    numwinners : int, optional
        Number of front-runners to retrieve. The default is 2.
    kind : str
        Front runner determination strategy with options:
        
        - 'tally' - Use election method determined tally system. Falls back to 
          'elimination' if 'tally' output not found.   
          
        - 'elimination' - Find the election winner, then eliminate the winner
          and determine the runner-up if the winner did not exist. 
          
        - 'score' - Determine tally using scored voting. 
        
        - 'plurality' - Determine tally using plurality voting
        
        - 'regret' - Determine tally using voter regret.
        
    erunner : :class:`~votesim.votemethods.eRunner` or None
        Optional, If an election runner is available, you can input it here 
        to reduce computation cost. 
          
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
    
    if kind == 'score':
        etype = 'score'
        kind = 'tally'
    elif kind == 'plurality':
        etype = 'plurality'
        kind = 'tally'
        
    frunners = []
    #ballots = ballots.copy()
    
    if erunner is not None:
        er = erunner
    elif election is not None:
        er = election.result.runner
        estats = election.electionStats
    else:
        er = ballots.run(etype, numwinners=1)
        
    if kind == 'tally':    
        try:
            # Some voting systems have an output called 'talley' which returns
            # tallys for each candidate that determine the winner. 
            # For these systems use tally to get the frontrunners. 
            tally = er.output['tally']
            winners, ties = votemethods.tools.winner_check(tally,
                                                           numwin=numwinners)
            return np.append(winners, ties)
        
        except (KeyError, TypeError):
            pass
        
    elif kind == 'regret':
        tally = regret_tally(estats)
    
    # If tally doesn't exist, determine front runner by rerunning
    # the election without the winner.
    while len(frunners) < numwinners:

        winners = er.winners
        ties = er.ties

        if len(winners) > 0:
            frunners.extend(winners)
            ballots.ratings[:, winners] = 0
            ballots.ranks[:, winners] = 0
            ballots.votes[:, winners] = 0
            
        elif len(ties) > 0:
            frunners.extend(ties)
            ballots.ratings[:, winners] = 0
            ballots.ranks[:, winners] = 0
            ballots.votes[:, winners] = 0            
        else:
            raise ValueError('No ties or winners found for election method.')
            
        er = ballots.run(etype, numwinners=1)
            
    return np.array(frunners)



def additional_tally_frontrunners(tally: np.ndarray, tol: float=.1):
    """Additional potential front runners that are not 1st and 2nd.
    

    Parameters
    ----------
    tally : numpy.ndarray(cnum,)
        Some sort of tally of a metric which describes how much a candidate
        is probably winning an election. 
    tol : float, optional
        Threshold at which to consider candidate as a front runner 
        from [0 to 1]. The default is .05.

    Returns
    -------
    out : numpy.ndarray(fnum,)
        Alternative Front runners that are not 1st and 2nd place. 
    """
    if tol == 0:
        return np.array([], dtype=int)
    
    # Get first place front runner
    tally = np.copy(tally)
    ii_1st = np.argmax(tally)
    tally_1st = tally[ii_1st]
    
    
    delta = tally_1st * tol
    cutoff = tally_1st - delta
    
    tally[ii_1st] = 0
    ii_2nd = np.argmax(tally)
    tally[ii_2nd] = 0
    
    # Get alternative front runners that meet the cutoff. 
    return np.where(tally >= cutoff)[0]
    
