# -*- coding: utf-8 -*-
"""
Test every voting method.
"""
import pdb 
import logging

import numpy as np
import votesim
from votesim import votemethods
from votesim.votemethods import scored_methods, ranked_methods, eRunner
from votesim.models import spatial


def test_all():
    numvoters = 50    
    num_candidates = 6
    numwinners_list = [1, 3, 5]
    
    for ii in range(50):
        v = spatial.Voters(seed=ii,)
        v.add_random(numvoters=numvoters, ndim=2, )
        c = spatial.Candidates(voters=v, seed=0)
        c.add_random(cnum=num_candidates, sdev=1.0)
        e = spatial.Election(voters=v, candidates=c)
        
        scores = e.ballotgen.get_honest_ballots(
            etype=votesim.votemethods.SCORE
        )
        ranks = e.ballotgen.get_honest_ballots(
            etype=votesim.votemethods.IRV
        )       
        
        for etype in ranked_methods:
            for numwinners in numwinners_list:
                runner = eRunner(etype=etype,
                                 ballots=ranks, 
                                 numwinners=numwinners)
                winners = np.unique(runner.winners)
                assert len(winners) == numwinners, etype

        for etype in scored_methods:
            for numwinners in numwinners_list:
                runner = eRunner(etype=etype,
                                 ballots=scores,
                                 numwinners=numwinners) 
                
                winners = np.unique(runner.winners)
                assert len(winners) == numwinners, etype
                
            
if __name__ == '__main__':
    # logging.basicConfig()
    # logger = logging.getLogger('votesim.votemethods.irv')
    # logger.setLevel(logging.DEBUG)
    
    test_all()
