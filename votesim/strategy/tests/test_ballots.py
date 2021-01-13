"""
Test error generation that error magnitude is indeed growing
"""
# -*- coding: utf-8 -*-

import numpy as np
from votesim.models import spatial
from votesim import ballot
# from votesim.strategy import TacticalRoot, Ra
from votesim.strategy.tacticalballots import (TacticalRoot,
                                              TacticalGroup,
                                              RatedTactics,
                                              RankedTactics,
                                              StrategyData)

etype = 'score5'
vnum = 8
rs = np.random.RandomState(0)
distances = rs.rand(vnum, 3)
# tol = np.ones((20, 3)) * 0.75
tol = None
b = ballot.gen_honest_ballots(distances=distances, tol=tol,
                              base='linear')
ballots = b.ratings


## Test compromise strategy
# def test_compromise():
#     bt = TacticalBallots(etype=etype, ballots=b)
#     bt.set()
#     bt = bt.compromise()
    
#     ii = np.arange(len(bt.ratings))
#     jj = bt.preferred_frontrunner
    
#     assert np.all(bt.ratings[ii, jj] == 1)
#     assert np.all(bt.ranks[ii, jj] == 1)

strategy = StrategyData({})
    
    
def test_compromise_rated():
    
    ballots = b.ratings
    root = TacticalRoot(etype, ballots=ballots, distances=distances)
    tgroup = root.get_tactical_group(strategy=strategy)
    b_rated = RatedTactics(ballots, group=tgroup)
    b_rated.compromise()
    ii = np.arange(len(ballots))
    jj = tgroup.preferred_frontrunner
    assert np.all(b_rated.ballots[ii, jj] == 1)

    
def test_compromise_ranked():
    
    ballots = b.ranks
    root = TacticalRoot(etype, ballots=ballots, distances=distances)
    tgroup = root.get_tactical_group(strategy=strategy)
    b_ranked = RankedTactics(ballots, group=tgroup)
    b_ranked.compromise()
    ii = np.arange(len(ballots))
    jj = tgroup.preferred_frontrunner    
    assert np.all(b_ranked.ballots[ii, jj] == 1)
    
    
def test_compromise_rated2():
    strategy = {'tactics' : ['compromise']}
    
    ballots = b.ratings
    root = TacticalRoot(etype, ballots=ballots, distances=distances)
    tgroup = root.get_tactical_group(strategy=strategy)
    new = root.modify_ballot(ballots=ballots, strategy=strategy)

    ii = np.arange(len(ballots))
    jj = tgroup.preferred_frontrunner
    assert np.all(new[ii, jj] == 1)    
    
if __name__ == '__main__':
    test_compromise_rated()
    test_compromise_rated2()
    test_compromise_ranked()