"""
Base data classes that are used by models, in an attempt
to keep data relatively immutable. 
"""
# -*- coding: utf-8 -*-
from dataclasses import dataclass, replace
import numpy as np
import votesim
from typing import NamedTuple



class VoteSimData:
    """Base data class"""
    def replace(self, **kwargs):
        return replace(self, **kwargs)
    

@dataclass(frozen=True)
class VoterData(VoteSimData):
    """Voter data storage class. 
    
    Attributes
    ----------
    strategy : dict
        Strategic parameters for voters
    pref : array (v, ndim)
        Preferences for `v` number of voters in `ndim` # of dimensions.
    weights : None or array (v, ndim)
        Preference weighting for `v` voters in `ndim` # of dimensions. 
    order : int
        Order or norm calculation for voter-candidate regret distance.
    stats : `votesim.metrics.VoterStats`
        Repository of voter statistics output.
    tol : float or None
        Voter preference max tolerance.
    base : str
        Voter rating mapping to distance, either:
            - 'linear' - Linear mapping of distance to rating
            - 'quadratic' - Quadratic mapping of distance to rating
            - 'sqrt' - Square root mappiong of distance to rating
            
        
    """
    # ballot_params: dict

    pref: np.ndarray 
    weights: np.ndarray 
    order: int 
    stats: "votesim.metrics.VoterStats" 
    tol: float=None
    base: str='linear'

    
@dataclass(frozen=True)
class VoterGroupData(VoteSimData):
    """Grouped voter data storage class for multiple `Voters`.
    
    Attributes
    ----------
    groups : tuple[`VoterData`]
        Groups of `VoterData`
    pref : array (v, ndim)
        Preferences for `v` number of voters in `ndim` # of dimensions.
    stats : `votesim.metrics.VoterStats`
        Voter statistics output.          
    group_index : dict[int, ndarray]
        Index locations for each group in `groups`.
    weights : None or array (v, ndim)
        Preference weighting for `v` voters in `ndim` # of dimensions. 
    order : int
        Order or norm calculation for voter-candidate regret distance.

    """
    groups : tuple
    pref: np.ndarray
    stats: "votesim.metrics.VoterStats"
    group_index: dict
    weights: np.ndarray = None
    order: int = None    
    

@dataclass(frozen=True)
class CandidateData(VoteSimData):
    """Model candidate data.
    
    Attributes
    ----------
    pref : array (c, ndim)
        Preferences for `c` number of candidates in `ndim` # of dimensions.
    distances : array of (v, c)
        Regret distance of each voter (rows) for each candidate (columns).
    stats : `votesim.metrics.CandidateStats`
        Candidate statistics
    """
    pref: np.ndarray
    distances: np.ndarray
    stats: "votesim.metrics.CandidateStats"



class StrategyData(NamedTuple):
    """Model strategy data.
    
    Attributes
    ----------
    tactics : tuple[str]
        List of tactics to apply
    subset : str
        Tactical subset of voters this strategy applies to. Set to either 
        
        - ''
        - 'topdog'
        - 'underdog'
        
    ratio : float
        Ratio of tactical voteres in this group from [0 to 1].
    index : slice
        Index locations of voters to consider
    underdog : int or None
        Use this to explicitly set an underdog to tactically vote in favor of. 
    groupnum : int
        Group index number corresponding to group index in
        `votesim.spatial.VoterGroup`.        
    frontrunnertype : str
        Method to calculate frontrunner.
    frontrunnernum : int
        Base number of front runners to consider.
    frontrunnertol : float
        Ratio of tally for consideration of additional front runners
    """    
    # These are "strategic" parameters based on front runner determination.
    tactics: tuple=()
    subset: str=''

    ratio: float=1.0
    index : slice=slice(None)
    underdog: int = None
    groupnum: int = None
    
    frontrunnertype: str='eliminate'
    frontrunnernum: int=2
    frontrunnertol: float=0.0

    # These are "honest" monotonic parameters
    # tol: float=None
    # base: str='linear'
    

def strategy_data(d):
    """Create and pass along StrategyData."""
    try:
        return StrategyData(**d)
    except TypeError:
        return d



    
@dataclass(frozen=True)
class ElectionData(VoteSimData):
    """Model Election data.
    
    Attributes
    ----------
    ballots : array (v, c)
        Voter ballot results for `v` voters and `c` candidates. 
    winners : array ( w,)
        Winning candidate index locations for `w` # of winners.
    ties : array (t,)
        Tied candidates of `t` total ties. 
    group_index : dict of array(j,)
        Dict indexed by group name, dict entries return the group's voters'
        index location. 
    
    """
    # voters: VoterData
    # candidates: CandidateData
    # distances: np.ndarray
    ballots: np.ndarray
    winners: np.ndarray
    ties: np.ndarray
    group_index: dict

    
@dataclass
class ElectionResult(object):
    """Data subclass constructed by `votesim.spatial.Election`. 
    Election result data stored here.
    
    Attributes
    ----------
    winners : array (w,)
        Winning candidate index locations for `w` # of winners.
    ties : array (t,)
        Tied candidate index locations for `t` # of ties. 
    ballots : array (v, c)
        Voter ballot results for `v` voters and `c` candidates. 
    runner : `votesim.votemethods.eRunner`
        Election runner output
    output : dict
        Election dictionary output
    output_docs : dict
        Election dictionary output documentation & descriptions. 
    stats : `votesim.metrics.ElectionStat`
        Election output statistics
    scoremax : int or None
        Max score for scored ballots.
    """
    winners: np.ndarray
    ties: np.ndarray
    ballots: np.ndarray
    
    runner : votesim.votemethods.eRunner
    output: dict
    output_docs: dict
    stats: 'votesim.metrics.ElectionStats'
    scoremax : int=None
    


    