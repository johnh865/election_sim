# -*- coding: utf-8 -*-
"""
Test plurality strategy

Assum is that strategy is voter behavior where voters predict the future
in order to change the results in their favor. 

"""
import numpy as np
import votesim
from votesim.simulation import Election

class StrategicFaction(object):
    """Construct a faction of strategic voters from a voter population
    
    A strategic faction is a spheroid conspiracy of voters who agree to vote in unison
    in order to maximize their satisfaction. 
    
    Parameters
    ------------
    election : votesim.simulation.Election
        A faction takes an eletion and develops a voting strategy in reaction 
        to this election 
    coords : array shaped (ndim,)
        Centroid of faction
    width : float
        Width of faction
    prob : float
        Probability from [0 to 1] that voters in radius join the faction
    """
    
    def __init__(self, election : Election, coords, width, prob):
        self.coords = coords 
        self.width = width
        self.prob = prob
        self.election = election
        
        
    def get_voter_index(self):
        v = self.election.voters
        index = np.abs(v - self.coords) < self.width
        return index
        
        
    
