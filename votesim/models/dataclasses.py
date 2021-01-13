# -*- coding: utf-8 -*-
from dataclasses import dataclass, replace
import numpy as np
import votesim
from typing import NamedTuple



class VoteSimData:
    
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
        Repository of voter statistics
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
    strategy : dict
        Strategic parameters for voters
    pref : array (v, ndim)
        Preferences for `v` number of voters in `ndim` # of dimensions.
    weights : None or array (v, ndim)
        Preference weighting for `v` voters in `ndim` # of dimensions. 
    order : int
        Order or norm calculation for voter-candidate regret distance.
    stats : `votesim.metrics.VoterStats`
        Voter statistics    
    """
    groups : tuple
    pref: np.ndarray
    stats: "votesim.metrics.VoterStats"
    group_index: dict
    weights: np.ndarray = None
    order: int = None    
    

@dataclass(frozen=True)
class CandidateData(VoteSimData):
    """Candidate data storage.
    
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
    """
    Attributes
    ----------
    tactics : tuple[str]
        List of tactics to apply
    subset : str
        Tactical subset of voters this strategy applies to. Set to either 
        
        - ''
        - 'topdog'
        - 'underdog'
    frontrunnertype : str
        Method to calculate frontrunner
    frontrunnernum : int
        Base number of front runners to consider
    frontrunnertol : float
        Ratio of tally for consideration of additional front runners
    index : slice
        Index locations of voters to consider
    underdog : int or None
        Use this to specifically set an underdog. 
        
    """
    
    # These are "strategic" parameters based on front runner determination.
    tactics: tuple=()
    subset: str=''
    frontrunnertype: str='tally'
    frontrunnernum: int=2
    frontrunnertol: float=0.0
    ratio: float=1.0
    index : slice=slice(None)
    underdog: int = None
    groupnum: int = None
    
    # These are "honest" monotonic parameters
    # tol: float=None
    # base: str='linear'
    

def strategy_data(d):
    """Create of pass along StrategyData."""
    try:
        return StrategyData(**d)
    except TypeError:
        return d



    
@dataclass(frozen=True)
class ElectionData(VoteSimData):
    """Election data storage.
    
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
    ballots : array (v, c)
        Voter ballot results for `v` voters and `c` candidates. 
    runner : `votesim.votemethods.eRunner`
        Election runner output
    output : dict
        Election dictionary output
    docs : dict
        Election dictionary output documentation & descriptions. 
    electionStats : `votesim.metrics.ElectionStat`
        Election output statistics
    """
    winners: np.ndarray
    ties: np.ndarray
    ballots: np.ndarray
    
    runner : votesim.votemethods.eRunner
    output: dict
    output_docs: dict
    stats: 'votesim.metrics.ElectionStats'
    scoremax : int=None
    

@dataclass
class ResultOptions(object):
    voters : bool
    candidates : bool
    winner : bool
    
    
    