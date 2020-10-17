# -*- coding: utf-8 -*-
from dataclasses import dataclass, replace
import numpy as np
import votesim




class VoteSimData:
    
    def replace(self, **kwargs):
        return replace(self, **kwargs)
    

@dataclass(frozen=True)
class VoterData(VoteSimData):
    strategy: dict
    pref: np.ndarray 
    weights: np.ndarray 
    order: int 
    voterStats: "votesim.metrics.VoterStats" 
    
    
@dataclass(frozen=True)
class VoterGroupData(VoteSimData):
    pref: np.ndarray
    voterStats: "votesim.metrics.VoterStats"
    weights: np.ndarray = None
    order: int = None    
    groups : tuple = ()
    

@dataclass(frozen=True)
class CandidateData(VoteSimData):
    pref: np.ndarray
    distances: np.ndarray

    
@dataclass(frozen=True)
class ElectionData(VoteSimData):
    # voters: VoterData
    # candidates: CandidateData
    # distances: np.ndarray
    ballots: np.ndarray
    winners: np.ndarray
    ties: np.ndarray
    
    
@dataclass
class ElectionResult(object):
    """Subclass constructed by `Election`. Election result data stored here.
    
    Attributes
    ----------
    winners : array (a,)
    
    ballots : array (v, a)
    
    runner : `votesim.votemethods.eRunner`
    
    output : dict
    
    docs : dict
    
    electionStats :     
    """
    winners: np.ndarray
    ties: np.ndarray
    ballots: np.ndarray
    
    runner : votesim.votemethods.eRunner
    output: dict
    output_docs: dict
    electionStats: 'votesim.metrics.ElectionStats'
    


    
if __name__ == '__main__':
    v = VoterData(strategy = {})
    v.replace()