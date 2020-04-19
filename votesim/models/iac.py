# -*- coding: utf-8 -*-

"""
Impartial Anonymous Culture Model
"""
import numpy as np
from votesim.models.spatial.base import _RandomState
from votesim.models.spatial.base import Election as _SpatialElection
from votesim.models import vcalcs
from votesim import utilities
VOTERS_BASE_SEED = 100

class Voters(object):
    def __init__(self, numvoters, seed=None, strategy='abs', stol=1.0):
        self.init(numvoters, seed, strategy, stol)
        return
    
    
    @utilities.recorder.record_actions(replace=True)
    def init(self, numvoters, seed, strategy, stol):
        self.numvoters=numvoters
        self.seed = seed
        self.strategy = strategy
        self.stol = stol
        self._randomstate = _RandomState(seed, VOTERS_BASE_SEED)  
        return      

    
    def calc_ratings(self, candidates):
        """
        Calculate preference distances & candidate ratings for a given set of candidates
        """
        rs = self._randomstate
        distances = rs.rand(self.numvoters, candidates)        
        ratings = vcalcs.voter_scores_by_tolerance(
                                                   voters=None,
                                                   candidates=None,
                                                   distances=distances,
                                                   tol=self.stol,
                                                   cnum=None,
                                                   strategy=self.strategy,
                                                   )
        self.ratings = ratings
        self.distances = distances
        return ratings
    
    

class Candidates(object):
    def __init__(self, cnum):
        self.init(cnum)
        return
        
        
    @utilities.recorder.record_actions(replace=True)
    def init(self, cnum):
        self.cnum=cnum
        self.candidates = cnum
        return
    

class ElectionStats(object):
    def __init__(self, distances):
        self.distances = distances
        return
    

    @utilities.lazy_property2('_cache_candidate')
    def candidate_regrets(self):
        """array shape (c) : voter regret for each candidate"""

        regrets = np.mean(self.distances, axis=0)
        return regrets
    
    
    
    
    
    

class Election(_SpatialElection):
    def _get_stats(self):
        """Retrieve election statistics and post-process calcualtions"""
        # TODO 
        
        distances = self.voters.distances
        
        stats = self.voters.ElectionStats
        stats.set(candidates=self.candidates.candidates,
                  winners=self.winners,
                  ballots=self.ballots)
    
        
        ### Build dictionary of all arguments and results 
        results = {}        
        results.update(self.get_parameters())
        
        results['output'] = stats.get_dict()
        results['output']['ties'] = self.ties
        
        results = utilities.misc.flatten_dict(results, sep='.')
        self.results = results        
        
        self._result_history.append(results)
        return results    
    

    


if __name__ == '__main__':
    v = Voters(60, 0)
    c = Candidates(6)
    e = Election(v, c)
    