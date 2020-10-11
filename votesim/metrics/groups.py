# -*- coding: utf-8 -*-
import pdb
import numpy as np
from typing import List

from votesim import utilities
from votesim.models import vcalcs

from votesim.metrics.metrics import (ElectionData,
                                     BaseStats,
                                     ElectionStats,
                                     )


class GroupStatsBase(BaseStats):
    """
    Parameters
    ----------
    electionStats : `ElectionStats`
        ElectionStats parent object
    group_indices : dict of arrays
        Voter index locations for each group
    winners : list of int
        Winning candidate index

    Attributes
    ----------
    _electionStats : ElectionStats
        Top-level output object
    _electionData : ElectionData
        Temporary election data used for making calculations
    _name : str
        Name of statistic for output dict
    _winners : list of int
        Winners
    _indices : dict of arrays
        Group_indices
    """    
    
    def __init__(self,
                 electionStats: ElectionStats,
                 group_indices: dict,
                 winners: List[int]
                 ):
        self._electionStats = electionStats
        self._electionData = electionStats.electionData
        self._indices = group_indices
        self._winners = winners
        self._reinit()
        return
    
    
    def _reinit(self):
        # self._indices = self._electionData.group_indices
        self._group_names = self._indices.keys()
        self._distances = self._electionData.distances
        # self._winners = self._electionData.winners
        return

    
    @utilities.lazy_property
    def _regret(self):
        """array(ngroups,) : Regret for each group."""
        distances = self._distances[:, self._winners]
        regrets = []
        for key, index in self._indices.items():
            d = distances[index].mean(axis=1).mean()
            regrets.append(d)
            
        return np.array(regrets)
    
    
    @utilities.lazy_property
    def regret(self):
        """dict[float] : Regret for each group, indexed by group name.""" 
        return dict(zip(self._group_names, self._regret))


    @utilities.lazy_property
    def regret_efficiency_candidate(self):
        """Voter satisfaction efficiency, compared to random candidate."""
        random = self._electionStats.candidate.regret_avg
        best = self._electionStats.candidate.regret_best

        U = self._regret
        R = random
        B = best
        vse = (U - R) / (B - R)
        return dict(zip(self._group_names, vse))


    @utilities.lazy_property
    def regret_efficiency_voter(self):
        """Voter satisfaction.
        
        VSE equation normalized to voter 
        population regret of an ideal winner vs a random voter.
        """
        v_random = self._electionStats.voter.regret_random_avg
        v_median = self._electionStats.voter.regret_median
        best = self._electionStats.candidate.regret_best

        U = self._regret
        R2 = v_random
        R1 = v_median
        B = best
        out = 1.0 - abs(U - B) / (R2 - R1)
        return dict(zip(self._group_names, out))    
    
    
class GroupStats(GroupStatsBase):
    """Regrets for voter groups as well as one-sided tactical groups."""
    def __init__(self, electionStats: ElectionStats):
        self._electionStats = electionStats
        self._electionData = electionStats.electionData
        self._indices = self._electionData.group_indices
        self._winners = self._electionData.winners
        self._reinit()
        self._name = 'group'
        
        return

    
class TacticCompare(GroupStatsBase):
    """For each group compare reffect of strategy vs honesty on regret"""
    def __init__(self, e_strat: ElectionStats, e_honest: ElectionStats):
        
        group_strate = GroupStats(e_strat)
        indices = group_strate._indices
        winners_honest = e_honest.electionData.winners
        group_honest = GroupStatsBase(electionStats=e_strat, 
                                      group_indices=indices,
                                      winners=winners_honest)
        self._group_strate = group_strate
        self._group_honest = group_honest
        self._electionStats = group_strate._electionStats
        self._electionData = group_strate._electionData
        self._name = 'tactic_compare'
        self._group_names = group_strate._indices.keys()
        return

        
    @utilities.lazy_property
    def _regret(self):
        r_strate = self._group_strate._regret
        r_honest = self._group_honest._regret
        return r_strate - r_honest
    
        
    @utilities.lazy_property
    def regret_efficiency_candidate(self):
        """Change in Voter satisfaction efficiency, tactical - honest."""
        names = self._group_names
        d = {}
        for name in names:
            v_strate = self._group_strate.regret_efficiency_candidate[name]
            v_honest= self._group_honest.regret_efficiency_candidate[name]
            d[name] = v_strate - v_honest
        return d



    @utilities.lazy_property
    def regret_efficiency_voter(self):
        """Change in Voter satisfaction, tactical - honest.
        
        VSE equation normalized to voter 
        population regret of an ideal winner vs a random voter.
        """
        names = self._group_names
        d = {}
        for name in names:
            v_strate = self._group_strate.regret_efficiency_voter[name]
            v_honest= self._group_honest.regret_efficiency_voter[name]
            d[name] = v_strate - v_honest
        return d




