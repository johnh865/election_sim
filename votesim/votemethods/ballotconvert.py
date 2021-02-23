# -*- coding: utf-8 -*-

import numpy as np
from typing import List, Dict


def convert_single_votes(votes : List[Dict[str, float]]):
    """Convert list of dict voter ballots to voter data used in this package.
    
    Parameters
    ----------
    votes : list[dict[str, object]]
        list[`v`] of dict for `v` voter. 
        
        Each dict key is a candidate and the associated ballot rank/value/score. 
    """
    candiate_dict = {}
    candidate_set = set()
    for ballot in votes:
        candidates = ballot.keys()
        
        
        
        
    
    