# -*- coding: utf-8 -*-
"""

Voter spatials models with variations of voter behavior of

1. Voter Error -- Voters with error in regret/distance calculation

2. Voter Ignorance -- Voters with limited memory and will only evaluate a
   finite number of candidates.

3. Min/Max voters -- Voters who min/max their scored ballot and do not rank all
   candidates
   
3. Bullet voters -- Voters who only vote for the top % of candidates. 


"""
import numpy as np

from votesim.models.spatial import Voters
from votesim.models import vcalcs
from votesim import utilities

class ErrorVoters(Voters):
    """Voters who get things wrong"""


    @utilities.recorder.record_actions()
    def add_random(self,
                   numvoters,
                   ndim=1, 
                   error_mean=0.0, 
                   error_width=0.0,
                   clim_mean=-1,
                   clim_width=2):
        """Add random normal distribution of voters
        
        Parameters
        -----------
        numvoters : int
            Number of voters to generate
        ndim : int
            Number of preference dimensions of population 
        error_mean : float
            Average error center of population
            
            - At 0, half population is 100% accurate 
            - At X, the the mean voter's accuracy is X std-deviations of 
              voter preference,
            
        error_width : float
            Error variance about the error_mean
            
        """    
        super(ErrorVoters, self).add_random(numvoters, ndim=ndim)
        self._add_error(numvoters,
                        error_mean=error_mean,
                        error_width=error_width)
        self._add_ignorance(numvoters, clim_mean, clim_width)
        return    
    
    
    @utilities.recorder.record_actions()
    def add_points(self, 
                   avgnum,
                   pnum,
                   ndim=1, 
                   error_mean=0.0,
                   error_width=0.0,
                   clim_mean=-1,
                   clim_width=2):
        """Add a random point with several clone voters at that point
        
        Parameters
        -----------
        avgnum : int
            Number of voters per unique point
        pnum : int
            Number of unique points
        ndim : int
            Number of dimensions
            
        """
        vnum1 = len(self.voters)
        super(ErrorVoters, self).add_points(avgnum, pnum, ndim=ndim)
        vnum2 = len(self.voters)
        vdiff = vnum2 - vnum1        
        self._add_error(vdiff,
                        error_mean=error_mean,
                        error_width=error_width)
        self._add_ignorance(vdiff, clim_mean, clim_width)
        
        return
    
           

    
    def _add_error(self, numvoters, error_mean=0.0, error_width=0.0):
        """Create voter error attribute for the specified number of voters
        self.voter_error describes the maximum candidate distance error
        a voter will make during the election.
        """
        
        rs = self._randomstate
        e = rs.normal(loc=error_mean, 
                      scale=error_width,
                      size=(numvoters,))
        
        e = np.maximum(0, e)
        try:
            error = np.concatenate((self.voter_error, e))
        except AttributeError:
            error = e
        
        self.voter_error = error
        return
    
    
    def _add_ignorance(self, numvoters, avg=7, std=2):
        rs = self._randomstate
        
        
        # if -1 thenn voters have perfect memory
        if avg == -1:
            cnum = np.ones(numvoters) * -1
        else:
            cnum = rs.normal(loc=avg, scale=std, size=(numvoters,))
            cnum = np.maximum(0, cnum)
        
        try:
            self.voter_memory = np.concatenate((self.voter_memory, cnum))
        except AttributeError:
            self.voter_memory = cnum
        return
    
    
    def calculate_distances(self, candidates):
        """Calculate regret distances.
        
        Parameters
        ----------
        candidates : array shaped (a, b)
            Candidate preference data
        """        
        pref = self.pref
        error = self.voter_error
        rs = self._randomstate        
        
        try:
            weights = self.weights
        except AttributeError:
            weights = None
            
        distances = vcalcs.voter_distances(voters=pref,
                                           candidates=candidates,
                                           weights=weights)
        distances = vcalcs.voter_distance_error(distances, error, rstate=rs)
        return distances    
    
