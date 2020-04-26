"""
Simulate strategic voters for point voter model
"""



# -*- coding: utf-8 -*-
import copy
import itertools

import numpy as np
import sympy
from sympy.utilities.iterables import multiset_permutations

import votesim

from votesim.models.spatial.base import SimpleVoters, Candidates, Election, _RandomState
from votesim import utilities

STRATEGIC_BASE_SEED = 5

def all_ranks(cnum):
    """Construct all possible permutations of rank ballosts, including 
    zeroing out candidates"""
    
    a = np.arange(1, cnum+1)
    
    new = []
    for p in multiset_permutations(a):
        new.append(p)
    return np.array(new)


def all_scores(cnum, maxscore):
    a = np.arange(0, maxscore+1)
    iter1 = itertools.product(a, repeat=cnum)
    new = [i for i in iter1]
    return np.array(new)


def random_scores(cnum, scoremax, num=1000, rs=None):
    """Generate random scored ballots
    """
    if rs is None:
        rs = np.random.RandomState(None)
        
    d = rs.rand(num, cnum) * 2 - 1.
    d[d < 0] = 0

    dimax = np.max(d, axis=1)
    dimax[dimax == 0] = 1
    d = d / dimax[:, None]
    
    d = d * scoremax
    d = np.round(d)
    
    zballot = np.zeros((1, cnum))
    d = np.vstack((d, zballot))     
    return d


def all_score_minmax(cnum, scoremax,):
    a = np.arange(0, 2)
    iter1 = itertools.product(a, repeat=cnum)
    new = [i for i in iter1]
    return np.array(new)
    


def random_ranks(cnum, num=1000, rs=None, allow_zeros=True):
    """Generate random ranked ballots
    """
    if rs is None:
        rs = np.random.RandomState(None)
        
    distances = rs.rand(num, cnum)    
    ranks = np.argsort(distances, axis=1) + 1
        
    if allow_zeros:
        threshold = rs.rand(num, 1) * cnum + 1
        ranks[ranks > threshold] = 0
    
    zballot = np.zeros((1, cnum))
    ranks = np.vstack((ranks, zballot))
    return ranks


def all_votes(cnum):
    """Generate every possible combination of ballots for single-mark ballot"""
    ballots = np.zeros((cnum + 1, cnum), dtype=int)
    
    for i in range(cnum):
        ballots[i, i] = 1
    return ballots


class StrategicElection(object):
    def __init__(self, election:Election, etype, seed=None):
        self.etype = etype
        self.election = election
        self._election_honest = copy.deepcopy(election)        
        self.init(seed=seed, etype=etype)
        return
    
    
    @property
    def election_honest(self):
        return self._election_honest
        
    
    
    @utilities.recorder.record_actions(replace=True)
    def init(self, seed, etype):
        """Initialize some election properties"""
        self._set_seed(seed)
        self.election_honest.run(etype)
        self.btype = self.election_honest.btype
        return    
    
    
    def _set_seed(self, seed):
        """ Set pseudorandom seed """
        if seed is None:
            self._seed = None
            self._randomstate = _RandomState(None)
        else:
            self._seed = (seed, STRATEGIC_BASE_SEED)
            self._randomstate = _RandomState(*self._seed)
        return


    @utilities.lazy_property
    def _voters_unique(self):
        u, inv, c =  np.unique(self.election.voters.voters, 
                               axis=0,
                               return_inverse=True,
                               return_counts=True)
        return u, inv, c
        
    @property    
    def voters_unique(self):
        """Retrieve unique voter coordinates"""
        return self._voters_unique[0]
    
    
    @utilities.lazy_property
    def group_indices(self):
        """
        list of int arrays shaped (a,)    
            Index locations of each group
        """
        unique_num = len(self.voters_unique)
        unique_locs = self._voters_unique[1]
        
        locations = []
        for i in range(unique_num):
            locs = np.where(unique_locs == i)[0]
            locations.append(locs)        
        return locations
    
    
    @property    
    def voters_unique_num(self):
        """Retrieve number of voters per unique voter group"""
        return self._voters_unique[2]    
    
    
    @utilities.lazy_property
    def ballot_combos(self):
        """Generate combinations of potential ballots"""
        e = self.election
        cnum = self.election.candidates.candidates.shape[0]
        bnum = 1000
        btype = e.btype
        scoremax = e.scoremax
        rs = self._randomstate

        if btype == 'rank':
            stratballots = random_ranks(cnum, num=bnum, rs=rs)
        
        elif btype == 'score':
            stratballots1 = random_scores(cnum, 
                                          num=bnum,
                                          scoremax=scoremax,
                                          rs=rs)
            stratballots2 = all_score_minmax(cnum, scoremax=scoremax)
            stratballots = np.vstack((stratballots1, stratballots2))
            
        elif btype == 'vote':
            stratballots = all_votes(cnum)

        return stratballots
    
    @utilities.lazy_property
    def ballot_combos_num(self):
        """Number of random ballot combinations generated for this election"""
        return len(self.ballot_combos)
    
    
    @utilities.lazy_property
    def honest_ballots(self):
        """Save honest ballots from the election here"""
        return self.election_honest.ballots.copy()
    
    
    @utilities.lazy_property
    def honest_regrets(self):
        """Save honest voter regrets for each voter group"""
        e = self.election_honest
        w = e.winners
        regrets = []
        for group, gindex in enumerate(self.group_indices):
            g_regret = e.voters.distances[gindex[0], w]
            regrets.append(g_regret)
            
        return np.array(regrets)
    
    
    def get_regrets(self):
        """Retrieve voter regrets for each voter group"""
        e = self.election_honest
        w = e.winners
        regrets = []
        for group, gindex in enumerate(self.group_indices):
            g_regret = e.voters.distances[gindex[0], w]
            regrets.append(g_regret)
            
        return np.array(regrets)
    
    
    
    
    def group_indices_strat(self, group, strat_ratio=1):
        """Retrieve ballot indexes for a voter group"""
        
        
        vindex = self.group_indices[group]
        imin = vindex.min()
        imax = vindex.max()
        inum = (imax - imin + 1)
        imax2 = imin + int(strat_ratio*inum)
            
        jj = (vindex <= imax2)
        
        return vindex[jj]
    
    
    
    def run_rand(self, strat_ratios=1, num_elections=5000):
        """
        Run random ballots.
        
        Find ballots that manage to result in a superior regret than an
        honest ballot. 
        """
        rs = self._randomstate
        
        groupnum = len(self.group_indices)
        s_ratios = np.ones(groupnum) * strat_ratios
        combos_num = self.ballot_combos_num
        
        
        
        # Obtain voter regrets for honest voting as benchmark
        
        
        
        # Get voter indices who will vote strategically
        group_locs = []
        for ii in range(groupnum):
            gii = self.group_indices_strat(ii, s_ratios[ii])
            group_locs.append(gii)
        
        # Select which ballots will be used for all elections
        combo_indices = rs.choice(combos_num, size=(num_elections, groupnum))
        
        for ii in range(num_elections):
            
            ballots = self.honest_ballots.copy()
            
            for jj in range(groupnum):
                cindex = combo_indices[ii, jj]
                ballot = self.ballot_combos[cindex]
                
                group_loc = group_locs[jj]
                ballots[group_loc] = ballot
                
            self.run(ballots)
            regrets 
            
                
                
            
        
        
        
        
        
        return
    
    
    def run(self, ballots):
        """re-run election with new ballots"""


        if self.btype == 'rank':
            self.election.ranks = ballots
        elif self.btype == 'score':
            self.election.scores = ballots
        elif self.btype == 'vote':
            self.election.votes = ballots
        
        self.election.run(etype=self.etype)
        return        
                
                
                
    @utilities.recorder.record_actions()
    def run_iter(self, voter_groups, strat_ratio):
        """
        Run strategic iteration of a single voter group. 
        
        Parameters
        -----------
        voter_group : int
            Index location of point voter group to strategize
        strat_ratio : float
            Ratio of strategic voters in voter group from [0.0 to 1.0]
        """
        newballots = self.honest_ballots.copy()
        
        
        vindexes = []
        vnums = []
        for voter_group in voter_groups:
            vindex = self.group_indices[voter_group]
            vnum = self.voters_unique_num[voter_group]
            vindexes.append(vindex)
            vnums.append(vnum)
            
            
            
            
            
        
        # Get honest results
        dnets = []
        winners = self.election.winners
        distances = self.election.voters.distances[vindex , winners] / vnum
        dnet = np.sum(distances)
        dnets.append(dnet)        
    
        for group in range(len(self.group_indices)):
            for ballot in self.ballot_combos:
                
                vindex = self.group_indices[group]
                newballots[vindex] = ballot
                
                if self.btype == 'rank':
                    self.election.ranks = newballots
                elif self.btype == 'score':
                    self.election.scores = newballots
                elif self.btype == 'vote':
                    self.election.votes = newballots
                
                self.election.run(etype=self.etype)
                winners = self.election.winners
                
                print(self.election.votes.sum(axis=0))
                print(winners)
                distances = self.election.voters.distances[vindex , winners] / vnum
                
                dnet = np.sum(distances)
                dnets.append(dnet)
            dnets = np.array(dnets)
            return dnets
            
                
                
            

        

    
    
    
    
        
        
