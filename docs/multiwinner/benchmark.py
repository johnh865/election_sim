# -*- coding: utf-8 -*-
import logging
import pdb
import numpy as np
import votesim
import votesim.benchmarks.runtools as runtools
from votesim.models import spatial
from votesim import votemethods
from votesim.metrics import TacticCompare
from votesim.models.vcalcs import voter_distances
from votesim.votemethods import plurality

 
logger = logging.getLogger(__name__)


def multi_model(name, methods, 
                 seed=0,
                 numvoters=201,
                 cnum=10,
                 num_parties=5,
                 num_winners=5,
                 num_districts=20,
                 party_discipline=0.0,
                 trialnum=1,
                 ndim=1,
                 stol=1,
                 base='linear'):
    """Simple Election model """
 
    e = spatial.Election(None, None,
                         seed=seed,
                         name=name,
                         numwinners=num_winners)
    
    v = spatial.Voters(seed=seed, tol=stol, base=base)
    v.add_random(numvoters, ndim=ndim)
    
    rs = np.random.RandomState(seed=(seed, 10))    
    center = np.zeros(ndim)
    parties = rs.normal(center, size=(num_parties, ndim), scale=1)
    candidate_prefs1 = rs.normal(
        center, 
        size=(cnum * num_districts, ndim),
        scale=1
        )
    candidate_party_dist = voter_distances(candidate_prefs1, parties)
    candidate_party_index = np.argmin(candidate_party_dist, axis=1)
    candidate_party_pref = parties[candidate_party_index]
    
    candidate_prefs = (
        candidate_prefs1 * (1 - party_discipline) +
        candidate_party_pref * party_discipline
        )
    candidate_prefs = candidate_prefs.reshape((num_districts, cnum, ndim))
    
    for cprefs in candidate_prefs:
        
        c = spatial.Candidates(v)
        c.add(cprefs)
        
        e.set_models(voters=v, candidates=c)
        for method in methods:
            e.run(etype=method)
        
        
        
        
        
    
    
    
    
    
    
    # cseed = seed * trialnum
    # for trial in range(trialnum):
    
    #     logger.debug(
    #         msg_base, 
    #         seed, numvoters, cnum, trial, ndim, stol, base)
        
    #     c = spatial.Candidates(v, seed=trial + cseed)
    #     c.add_random(cnum, sdev=1.5)
    #     e.set_models(voters=v, candidates=c)
        
    #     # Save parameters
    #     e.user_data(
    #                 num_voters=numvoters,
    #                 num_candidates=cnum,
    #                 num_dimensions=ndim,
    #                 voter_tolerance=stol
    #                 )
        
    #     for method in methods:
    #         e.run(etype=method)
            
    # return e



m = multi_model('test',
                 ['plurality'])