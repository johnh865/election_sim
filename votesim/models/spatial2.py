# -*- coding: utf-8 -*-

import collections
import pickle
import copy
from dataclasses import dataclass

from typing import List, NamedTuple, Tuple, Dict

import numpy as np
import pandas as pd
import scipy
from scipy.stats import truncnorm
import votesim
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
                                        VoteSimData
                                        )
from votesim.models.spatial import Voters
from votesim.utilities.math import ltruncnorm

@dataclass(frozen=True)
class VoterData2(VoteSimData):
    """Voter data storage class. 
    """
    # ballot_params: dict

    pref: np.ndarray 
    weights: np.ndarray 
    order: int 
    stats: "votesim.metrics.VoterStats" 
    tol: float=None
    base: str='linear'

class Voters2(Voters):
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
    def set_behavior(self, tol: float=None, base: str='linear', tol2: float=None):
        """Set voter strategy type."""
        self._tol = tol
        self._tol2 = tol2
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