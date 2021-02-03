# -*- coding: utf-8 -*-

"""
Impartial Anonymous Culture Model
"""
import numpy as np
from votesim.models import spatial
from votesim.models import vcalcs
from votesim import utilities, ballot
VOTERS_BASE_SEED = 100

raise NotImplementedError("This module is not ready.")




class Voters(spatial.Voters):
    def __init__(self, numvoters, seed=None, strategy='abs', stol=1.0):
        self.init(numvoters, seed, strategy, stol)
        return


    @utilities.recorder.record_actions()
    def add_random(self, numvoters):
        return super(self).add_random(numvoters, ndim=1)
        
    
    @utilities.recorder.record_actions()
    def add_points(self, avgnum, pnum):
        return super(self).add_points(avgnum,  pnum, ndim=1)
    
    
    def _add_voters(self, pref):
        pref = np.array(pref)
        assert pref.ndim == 2
        assert pref.shape[1] == 1
        return super(self)._add_voters(pref)
    
    
    def calculate_distances(self, candidates=None):
        return np.abs(self.pref)
    
    
    def honest_ballots(self, candidates=None):
        distances = self.calculate_distances()
        b = ballot.gen_honest_ballots(distances=distances,
                                       tol=self.strategy['tol'],
                                       base=self.strategy['base'])
        return b    
    

class Candidates(object):
    """A dummy class."""
    def __init__(self, voters: Voters):
        cnum = voters.pref.shape[1]
        self.pref = np.zeros((cnum, 1))
        self.voters = voters
                
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
    