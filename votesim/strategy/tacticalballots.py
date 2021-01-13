# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 16:59:30 2020

@author: John
"""
import warnings
import pdb
import functools
import logging
from dataclasses import dataclass
from typing import Tuple, NamedTuple
import numpy as np

import votesim
from votesim import votemethods
from votesim import utilities
from votesim.models.vcalcs import distance2rank
from votesim.metrics.metrics import regret_tally
from votesim.ballot import BaseBallots
from votesim.utilities.decorators import lazy_property
from votesim.models.dataclasses import (ElectionResult, 
                                        VoteSimData,
                                        VoterGroupData,
                                        StrategyData,
                                        strategy_data,)



    
logger = logging.getLogger(__name__)

# @dataclass

    

class TacticalRoot(object):
    """
    Parameters
    ----------
    etype : str
        Election method
    ballots : ndarray (v, c)
        Honest voter ballots to use as base. 
    result: ElectionResult
        Honest election results used for some calculations. If not availabe,
        must provide `distances`, and some options may not work.
    distances : ndarray (v, c)
        Optional, use if `result` not available. Voter regret distances 
        away from each candidate
    scoremax : int
        Optional, use if `result` not available. Maximum score for scored 
        ballots. 
        
    """
    def __init__(self, 
                 etype: str,
                 ballots: np.ndarray, 
                 result: ElectionResult=None,
                 distances: np.ndarray=None,
                 scoremax: int=None
                 ):
        
        self.etype = etype
        self.btype = votemethods.get_ballot_type(etype)
        self.ballots = ballots
        self.result = result
        self._tactical_groups = {}
        if result is None:
            self.distances = distances
            self.scoremax = scoremax
        else:
            
            self.distances = self.result.stats._candidate_data.distances
            self.scoremax = self.result.scoremax
            
        self._front_runner_gen = FrontRunnerGen(etype, 
                                                ballots=ballots,
                                                result=result)


    @functools.lru_cache
    def get_frontrunners(self, frontrunnertype, frontrunnernum, frontrunnertol):
        fg = self._front_runner_gen
        frunners = fg.get_frontrunners(kind=frontrunnertype,
                                      num=frontrunnernum, 
                                      tol=frontrunnertol)
        return frunners
    
    
    def get_honest_winners(self):
        return self._front_runner_gen.winners
    
    
    # @lazy_property
    # def distances(self):
    #     """Retrieve voter regret distances for each candidate."""
    #     return self.result.stats._candidate_data.distances
    
    
    def get_tactical_group(self, strategy: StrategyData):
        strategy = strategy_data(strategy)
        key = (str(strategy.index), strategy.subset)
        try:
            return self._tactical_groups[key]
        except KeyError:
            tg = TacticalGroup(self, strategy=strategy)
            self._tactical_groups[key] = tg
            return tg 

    
    def modify_ballot(self,
                      ballots: np.ndarray,
                      strategy: dict) -> np.ndarray:
        """Apply strategy to ballot. 

        Parameters
        ----------
        ballots : np.ndarray (v, c)
            Current ballots to modify for all voters and candidates
        strategies : list of dict
            Strategies to apply.
        index : slice
            Index locations to apply strategy onto voters. 

        Returns
        -------
        ballots : ndarray (v, c)
            Newly modified ballots for all voters and candidates.

        """
        btype = self.btype
        ballots = ballots.copy()
        
        strategy = strategy_data(strategy)
        
        # Construct tactical group info with voter index location
        tactical_group = self.get_tactical_group(strategy=strategy)
        
        # Construct tactical ballot constructor
        if btype == 'rank' or btype == 'vote':
            xtactics = RankedTactics(ballots, tactical_group)
            
        elif btype == 'rate':
            xtactics = RatedTactics(ballots, tactical_group)
        
        elif btype == 'score':
            ballots = ballots / self.scoremax
            xtactics = RatedTactics(ballots, tactical_group)
            

        # Chain through all tactics and call them. 
        tactics = strategy.tactics
        if isinstance(tactics, str):
            getattr(xtactics, tactics)()
        else:
            for name in tactics:
                getattr(xtactics, name)()
        ballots = xtactics.ballots
        
        # Apply score and vote ballot postprocessing
        if btype == 'score':
            ballots = np.round(self.scoremax * ballots)
        elif btype == 'vote': 
            ballots = votemethods.tools.getplurality(ranks=ballots)
        return ballots
    
    
    def apply_strategies(self, strategies: list):
        """Apply strategies to voter groups.
        
        Parameters
        ----------
        strategies : list of StrategyData
            Voter strategies
            
        Returns
        -------
        ballots : ndarray (v, c)
            New voter ballots
        """
        
        ballots = self.ballots
        for strategy in strategies:
            ballots = self.modify_ballot(ballots, strategy=strategy)

        return ballots
    
    
    def get_group_index(self, strategies: list):
        """Retrieve index locations of all tactical groupings for the given
        strategies. Note taht the groups change when strategy and voting
        methods change."""

        new = {}
        ii = 0 
        for strategy in strategies:
            tgroup = self.get_tactical_group(strategy=strategy)
            tdict = tgroup.index_dict
            
            for key, value in tdict.items():
                name = key + '-' + str(strategy.groupnum) 
                new[name] = value
            ii += 1
        return new
    
    
    def get_group_frontrunners(self, strategies: list):
        """
        Parameters
        ----------
        strategies : list
            Strategies
        
        Returns
        -------
        out : ndarray (a, 2)
            - Front runners for each strategy group
            - column 0 = Honest winner
            - column 1 = Front runner 
            
        """
        new = []
        ii = 0
        for ii, strategy in enumerate(strategies):
            tgroup = self.get_tactical_group(strategy=strategy)
            frunners = tgroup.front_runners
            new.append(frunners)
        return np.array(frunners)
            

    
class TacticalGroup(object):
    """Generate tactical information required to create voter strategies. 
    Retrieve front runners, underdog locations, and topdog locations. 
    TacticalGroup must be constructed using TacticalRoot. 
    
    Parameters
    ----------
    root : TacticalRoot
        Base for tactical generation
    strategy : StrategyData
        Strategy to apply onto the group
    
    Attributes
    ----------
    index_dict : dict of numpy.ndarray[int] size (ai,)
        Index locations of tactical subsets -- 
        
        - 'tactical-topdog' -- Voterse of topdog, honest winning candidate. 
        - 'tactical-underdog' -- Voters of underdog candidate.
        - 'tactical' -- Voters in this group who vote tactically 
        - 'honest' -- Voters in this group who vote honestly.
    
    index : ndarray (v1,)
        Index locations of voters in group.
    iloc_bool : ndarray (v2,)
        Boolean index locations of tactical voters in group. 
    iloc_int : ndarray (v2,)
        Integer index locations of tactical voters in group.
    iloc_int_all : ndarray(v,)
        Integer index locations of all voters of election. 
        
    """
    
    iloc_bool : np.ndarray
    iloc_int : np.ndarray
    iloc_int_all : np.ndarray
    index : np.ndarray
    
    def __init__(self,
                 root: TacticalRoot,
                 strategy: StrategyData):
        
        self.strategy = strategy
        self.root = root
        
        ratio = strategy.ratio
        index = strategy.index
            
        index_tactical = self._apply_ratio(index, ratio)
        
        self._get_index(index=index_tactical, subset=strategy.subset)
        self.distances = self.root.distances
        
        
    @lazy_property
    def front_runners(self):
        """Front runners for the provided ballots in `self.root`."""
        if self.strategy.underdog is not None:
            winners = self.root.get_honest_winners()
            new = np.append(winners, self.strategy.underdog)
            return new
        
        warnings.warn('Estimating front runner is sort of buggy and should be avoided..')
        frontrunnertype = self.strategy.frontrunnertype
        frontrunnernum = self.strategy.frontrunnernum
        frontrunnertol = self.strategy.frontrunnertol
        return self.root.get_frontrunners(frontrunnertype=frontrunnertype,
                                          frontrunnernum=frontrunnernum,
                                          frontrunnertol=frontrunnertol)
        
        
    def _apply_ratio(self, index, ratio):
        """Modify index using ratio of tactical voters.
        Construct new index for honest and tactical voters."""
        voter_num = len(self.root.ballots)
        
        # number of strategic voters for this group. 
        voter_num_strat = int(np.round(ratio * voter_num))
        slicei = index
        
        starti = slicei.start
        stopi = slicei.stop
        if starti is None and stopi is None:
            starti = 0
            stopi = voter_num
            endi = stopi
        else:
            endi = starti + voter_num_strat
        
        index_tactical = np.arange(starti, endi)
        # index_honest = np.arange(endi, stopi)
        
        return index_tactical
  

    def _get_index(self, index, subset=''):
        """Set enabled tactical voter index.
        
        Parameters
        ----------
        index : (a,) array
            Index locations of enabled tactical voters
        subset : str
            Tactical subset, either
                - '' = Use all of index
                - 'underdog = Only set underdog voters
                - 'topdog' = Only set topdog voters
        """      
        bnum = len(self.root.ballots)
        self.iloc_int_all = np.arange(bnum, dtype=int)
        
        if index is None:
            self.index = slice(None)
            #self._iloc_int = np.arange(bnum, dtype=int)
            self.iloc_bool = np.ones(bnum, dtype=bool)
        else:
            self.index = index
            self.iloc_bool = np.zeros(bnum, dtype=bool)
            self.iloc_bool[index] = True
            #self._iloc_int = np.where(self._iloc_bool)[0]
                        
        if subset == 'underdog':
            self.iloc_bool = self.iloc_bool_underdog & self.iloc_bool
        elif subset == 'topdog':
            self.iloc_bool = self.iloc_bool_topdog & self.iloc_bool
        elif subset == '':
            self.iloc_bool = self.iloc_bool.copy()
        else:
            raise ValueError(f'subset "{subset}" is not a valid value.')

        self.iloc_int = np.where(self.iloc_bool)[0]
        
        # Get honest locations
        self.iloc_int_honest = np.where(~self.iloc_bool)[0]
        return
    
    

    @utilities.lazy_property
    def _best_worst_frontrunner(self):
        frunners = self.front_runners
        distances = self.root.distances
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
        return np.argmax(self.root.distances, axis=1)

    
    @property
    def projected_winner(self):
        """Projected winner from honest votes."""
        return self.front_runners[0]
    
    
    @utilities.lazy_property
    def _index_under_top(self):
        """Calculate top dog and under dog index for all voters"""
        index = self.index
        
        bnum = len(self.root.ballots)
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
        """int array: Index locations of all underdog voters."""
        return self._index_under_top[0]
        
    
    @property
    def iloc_bool_topdog(self):
        """int array: Index locations of all topdog voters."""
        return self._index_under_top[1]
    
    
    @lazy_property
    def index_dict(self):
        """dict of numpy.ndarray[int] size (ai,)
            Index locations of tactical subsets -- 
        
        - 'topdog' -- Voters of topdog, honest winning candidate. 
        - 'underdog' -- Voters of underdog candidate.
        - 'tactical' -- Voters in this group who vote tactically 
        - 'honest' -- Voters in this group who vote honestly.
        """
        d = {}
        d['topdog'] = np.where(self.iloc_bool_topdog)[0]
        d['underdog'] = np.where(self.iloc_bool_underdog)[0]
        d['tactical'] = self.iloc_int
        d['honest'] = self.iloc_int_honest
        
        return d
        

    
                
class RatedTactics(object):
    """Apply rated ballot tactics on a TacticalGroup.
    
    Parameters
    ----------
    ballots : ndarray (v, c)
        Original ballots to modify.
    group : TacticalGroup
        The group to apply tactics on.
    """
    def __init__(self, ballots: np.ndarray, group: TacticalGroup):
        self.ballots = ballots.copy()
        self.group = group
        self.distances = self.group.distances
        self.iloc_bool = self.group.iloc_bool
        self.iloc_int = self.group.iloc_int
        self.iloc_int_all = self.group.iloc_int_all
        
        
        
    def compromise(self):
        """Maximize preference in favor of favorite front runner."""
        ii = self.group.iloc_int
        jj = self.group.preferred_frontrunner[self.group.iloc_int]
        self.ballots[ii, jj] = 1
        return self

    
    def bury(self):
        """Bury hated front-runner equal to most hated candidate"""
        ii = self.group.iloc_int
        jj = self.group.hated_candidate[ii]
        self.ballots[ii, jj] = 0
        return self
        
        
    def truncate_hated(self):
        """Truncate all candidates equal or worse than hated front-runner."""
        
        ii = self.group.iloc_int_all
        jj = self.group.hated_frontrunner
        dist_hated = self.distances[ii, jj]
        idelete  = self.distances >= dist_hated[:, None]
        idelete = idelete & self.group.iloc_bool[:, None]
        self.ballots[idelete] = 0
        return self
    
    
    def truncate_preferred(self):
        """Truncate all candidates worse than favorite frontrunner."""
        
        iall = self.group.iloc_int_all
        dist_fav = self.distances[iall, self.group.preferred_frontrunner]
        idelete = self.distances > dist_fav[:, None]
        idelete = idelete & self.group.iloc_bool[:, None]
        self.ballots[idelete] = 0
        return self
    

    def bullet_preferred(self):
        """Bullet vote for preferred front runner."""
        ii = self.group.iloc_int
        jj = self.group.preferred_frontrunner[self.group.iloc_int]
        
        self.ballots[ii, :] = 0
        self.ballots[ii, jj] = 1
        return self
    
    
    def bullet_favorite(self):
        """Bullet vote for your favorite candidate."""
        
        favorites = np.argmin(self.group.distances, axis=1)
        ii = self.group.iloc_int
        jj = favorites[ii]
        
        self.ballots[ii, :] = 0
        self.ballots[ii, jj] = 1
        return self
    
    
    def minmax_hated(self):
        """Max-min hated strategy.
        
        Zeroing all candidates equal or worse than 
        hated front-runner.
        """
        self.truncate_hated()
        ibool = self.group.iloc_bool[:, None]
        ij_bool = (self.ballots > 0) * ibool
        self.ballots[ij_bool] = 1.0
        return self
    
    
    def minmax_preferred(self):
        """Max-min favorite strategy.
        
        Zero all candidates equal or worse than 
        favorite front-runner.
        """
        self.truncate_preferred()
        ibool = self.group.iloc_bool[:, None]
        ij_bool = (self.ballots > 0) * ibool
        self.ballots[ij_bool] = 1.0
        return self
    
    
class RankedTactics(object):
    """Apply ranked ballot tactics on a TacticalGroup.
    
    Parameters
    ----------
    ballots : ndarray (v, c)
        Original ballots to modify.
    group : TacticalGroup
        The group to apply tactics on.
    """    
    def __init__(self, ballots: np.ndarray, group: TacticalGroup):
        self.ballots = ballots.copy()
        self.group = group
        self.distances = self.group.distances
    
        
    def compromise(self):
        """Maximize preference in favor of favorite front runner."""
        ii = self.group.iloc_int
        jj = self.group.preferred_frontrunner[self.group.iloc_int]
        
        ballots1 = self.ballots.astype(float)
        ballots1[ii, jj] = 0.5
        ballots1 = votemethods.tools.rcv_reorder(ballots1)
        self.ballots = ballots1
        return self
    
    
    def deep_bury(self):
        """Bury hated front-runner even lower than most hated candidate"""
        ii = self.group.iloc_int
        jj = self.group.hated_frontrunner[ii]
        self.ballots[ii, jj] = 0   
        self.ballots = votemethods.tools.rcv_reorder(self.ballots)
        return self
    
    
    def bury(self):
        """Bury hated front-runner equal to most hated candidate"""
        ii = self.group.iloc_int
        jj = self.group.hated_candidate[ii]
        self.ballots[ii, jj] = 0
        return self.deep_bury()
        
        
    def truncate_hated(self):
        """Truncate all candidates equal or worse than hated front-runner."""
        
        iall = self.group.iloc_int_all
        dist_hated = self.distances[iall, self.group.hated_frontrunner]
        idelete  = self.distances >= dist_hated[:, None]
        idelete = idelete & self.group.iloc_bool[:, None]
        
        self.ballots[idelete] = 0
        return self
    
    
    def truncate_preferred(self):
        """Truncate all candidates worse than favorite frontrunner."""
        
        iall = self.group.iloc_int_all
        dist_fav = self.distances[iall, self.group.preferred_frontrunner]
        idelete = self.distances > dist_fav[:, None]
        idelete = idelete & self.group.iloc_bool[:, None]

        self.ballots[idelete] = 0
        return self
        
        
    def bullet_preferred(self):
        """Bullet vote for preferred front runner."""
        ii = self.group.iloc_int
        jj = self.group.preferred_frontrunner[self.group.iloc_int]

        self.ballots[ii, :] = 0
        self.ballots[ii, jj] = 1
        return self
    
    
    def bullet_favorite(self):
        """Bullet vote for your favorite candidate."""
       
        favorites = np.argmin(self.distances, axis=1)
        ii = self.group.iloc_int
        jj = favorites[ii]
        self.ballots[ii, :] = 0
        self.ballots[ii, jj] = 1
        return self
  
             

class FrontRunnerGen(object):
    """Generate frontrunners (predicted winner and runner-up) of the election.
    
    Parameters
    ----------
    etype : str
        Election method to use.
    ballots : ndarray
        Voter input ballots 
    result : votesim.spatial.dataclasses.ElectionResult
        Election results if available. 
    
    """
    def __init__(self, 
                 etype: str,
                 ballots: np.ndarray=None, 
                 result: ElectionResult=None):
        
        self.etype = etype
        if ballots is not None:
            self.ballots = ballots
            self.runner = votemethods.eRunner(etype, ballots=ballots)
        elif result is not None:
            self.ballots = result.ballots
            self.runner = result.runner
            self.result = result
        else:
            raise ValueError('Either ballots or result must be specified.')
            
        self.winners = self.runner.winners
        return
    
    
    # @functools.lru_cache
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
        
        ballots = self.ballots.copy()
        numwinners = num
        frunners = []
        runner = self.runner

        while len(frunners) < numwinners:
            winners = runner.winners
            ties = runner.ties         
            
            if len(ties) > 0:
                frunners.extend(ties)
                ballots[:, ties] = 0
                
            elif len(winners) > 0:
                frunners.extend(winners)
                ballots[:, winners] = 0
         
            else:
                raise ValueError('No ties or winners found for election method.')
                
            runner = votemethods.eRunner(self.etype, ballots=ballots)
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
        else:
            raise ValueError(f'kind "{kind}" is not recognized tally type.')
        return tally
                
    
    def get_tally_method(self):
        """Retrieve tally generated by election method."""
        try:
            tally = self.runner.output['tally']
            return tally
        
        except (KeyError, TypeError):
            s = f'"tally" could not be found in voting method {self.etype} output.'
            raise TallyError(s)
                
    
    def get_tally_regret(self):
        """Retrieve tally calculated by voter regret."""
        estats = self.result.stats
        tally = regret_tally(estats)
        return tally
    
    
    def get_tally_condorcet(self):
        """Retrieve tally from a Condorcet-like process."""
        output = self.runner.output
        margins = output['margin_matrix']
        tally = np.min(margins, axis=0)
        worst = np.min(tally)
        return tally + worst


class TallyError(Exception):
    """Need a special error that can be caught by FrontRunners"""
    pass



    


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
    


class TacticalBallots(object):
    """Generate tactical ballots. 

    Parameters
    ----------
    etype : str
        Election method to use.
    strategies : list of StrategyData
        Voter strategies
    result : ElectionResult, optional
        Election results to form base ballots & strategic info.
        The default is None.
    ballots : np.ndarray, optional
        Mutually exclusive with `result`.
        If no results available, use this to set  basic initial ballots.
        The default is None.

    Attributes
    ----------
    ballots : np.ndarray
        Output strategic ballots.
    group_index : dict of ndarray
        Index locations of groups. 
    root : `TacticalRoot`
        ballot generation object. 
    """    
    
    def __init__(self,
                 etype: str, 
                 strategies: tuple,
                 result: ElectionResult=None, 
                 ballots: np.ndarray=None):     
        
        if ballots is None: 
            ballots = result.ballots
        
        root = TacticalRoot(etype=etype, ballots=ballots, result=result)
        ballots = root.apply_strategies(strategies)
        group_index = root.get_group_index(strategies)
        
        self.root = root
        self.ballots = ballots
        self.group_index = group_index
        return







            
def gen_tactical_ballots(etype: str, 
                         strategies: tuple,
                         ballots: np.ndarray,
                         result: ElectionResult=None, 
                         ):
    """Generate tactical ballots. 

    Parameters
    ----------
    etype : str
        DESCRIPTION.
    votergroup : VoterGroupData
        DESCRIPTION.
    result : ElectionResult, optional
        Election results to form base ballots & strategic info. The default is None.
    ballots : np.ndarray, optional
        If no results availabe, the basic initial ballots. The default is None.

    Returns
    -------
    ballots : np.ndarray
        Output strategic ballots.
    group_index : dict of ndarray
        Index locations of groups. 

    """

    if ballots is None: 
        ballots = result.ballots
    
    root = TacticalRoot(etype=etype, ballots=ballots, result=result)
    ballots = root.apply_strategies(strategies)
    group_index = root.get_group_index(strategies)
    return ballots, group_index

