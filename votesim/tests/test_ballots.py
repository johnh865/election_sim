# -*- coding: utf-8 -*-
import numpy as np
from votesim import ballot
import matplotlib.pyplot as plt


def test_rank_cut():
    """Test to make sure rank cutting is working."""
    np.random.seed(0)
    distances = np.random.rand(3, 5)
    ballots = ballot.BaseBallots(distances=distances, tol=0.5)
    ballots1 = ballots.rate_linear().rate_norm().rank_honest().rank_cut()
    zero_locs1 = ballots1.ratings == 0
    zero_locs2 = ballots1.ranks == 0
    assert np.all(zero_locs1 == zero_locs2)


def test_rating_shapes():
    """Test the 3 rating shapes linear, sqrt, and quadratic.
    The generated 
    """
    np.random.seed(0)
    distances = np.random.rand(100, 5)
    ballots = ballot.BaseBallots(distances=distances, tol=0.8)
    
    ballots1 = ballots.copy().rate_linear()
    ballots2 = ballots.copy().rate_sqrt()
    ballots3 = ballots.copy().rate_quadratic()
    
    plt.figure()
    plt.plot(ballots1.distances, ballots1.ratings, '.')
    plt.plot(ballots2.distances, ballots2.ratings, 'x')
    plt.plot(ballots3.distances, ballots3.ratings, 'o')
    plt.xlabel('Preference Distance')
    plt.ylabel('Ballot Score')
    plt.text(0.0, 0.5,
             "THIS PLOT SHOULD LOOK LIKE\n SQRT, LINEAR, AND QUADRATIC FUNCTION!")
    
    assert np.all(ballots1.ratings >= ballots3.ratings)
    assert np.all(ballots1.ratings <= ballots2.ratings)
    
    
    zero_locations = ballots1.ratings == ballots3.ratings
    assert np.all(ballots1.ratings[zero_locations] == 0)
    assert np.all(ballots2.ratings[zero_locations] == 0)
    assert np.all(ballots3.ratings[zero_locations] == 0)
    assert np.any(ballots1.ratings > 0)
    assert np.any(ballots2.ratings > 0)
    assert np.any(ballots3.ratings > 0)
    
    
if __name__ == '__main__':
    test_rank_cut()
    test_rating_shapes()