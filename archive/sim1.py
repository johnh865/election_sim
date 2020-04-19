# -*- coding: utf-8 -*-
"""
Simulate elections. 


Elements of an election

1. Create voter preferences
2. Create candidate preferences
3. Simulate voter behavior, strategy
4. Transform voter preferences into candidate scores or rankings
5. Input scores/ranks into election system.
6. Run the election.
7. Measure the results. 


"""
from votesystems import irv, plurality, score
import behavior

import numpy as np
import matplotlib.pyplot as plt






def estimate_rating(voters, candidates, tol=1):
    """
    Creating ratings from 0 to 1 of voter for candidates.
    
    - Assume that all voters are honest and have perfect information.
    - Assume single spectrum 1-dimensional political preferences for voters & politicians
    - Assume that voters have a preference "tolerance". Candidates whose preference
      distance exceeds this tolerance have utility set to zero. 
      - Linear mapping of preference distance and utility. 
      - Utility = 1 if candidate preference is exactly voter preference.
      - Utility = 0 if candidate preference is exaclty the tolerance distance. 
    - Assume that voters will give strongest possible preference to closest candidate,
      unless that candidate is outside their preference tolerance. 
    
    Parameters
    ----------
    voters : array shape (a, n)
        Voter preferences; n-dimensional voter cardinal preferences for n issues. 
    candidates : array shape (b, n)
        Candidate preferences for n-dimensional issues. 
    tol : float, or array shaped (a,)
        Voter candidate tolerance distance. If cardinal preference exceed tol, 
        utility set to zero. Toleranace is in same units as voters & candidates
    
    Returns
    -------
    out : array shaped (a, b) 
        Utility scores from a-voters for each b-candidate. 
    """
    
    # Create preference differences for every issue. 
    # diff = shape of (a, n, b) or (a, b)
    # distances = shape of (a, b)
    diff = voters[:, None] - candidates[None, :]
    if diff.ndim == 2:
        distances = np.abs(diff) 
    elif diff.ndim == 3:
        distances = np.linalg.norm(diff, axis=1)
    dmax = np.max(distances)
    dtol = dmax * tol
    
    
    i = np.where(distances > dtol)
    
    utility = (dtol - distances) / dtol
    utility[i] = 0
    
#    print(utility)
    max_utility = np.max(utility, axis=1)    
    i2 = np.where(max_utility == 0)
    max_utility[i2] = 1.
#    print(np.min(max_utility))
    return utility / max_utility[:, None]


def handle_ties(winners, ties, numwinners):
    """If ties are found, choose random candidate to break tie
    
    Parameters
    ----------
    winners : array shaped (a,)
        Winning candidate indices
    ties : array shaped (b,)
        Candidates that almost won but have received a tie with other candidates. 
    numwinners : int
        Total number of required winners for this election
    
    Returns
    --------
    winner : array shaped (numwinners,)
        Winners of election. Tie-broken by random choice. 
    """
    
    winners = np.array(winners)
    num_found = len(winners)
    num_needed =  numwinners - num_found
    
    if num_needed > 0:
        
        new = np.random.choice(ties, size=num_needed, replace=False)
        winners = np.append(winners, new)
    return winners.astype(int)


def simulate_election(voters, candidates, tol, numwinners, method, mtype, scoremax=5, kwargs=None):
    """
    Parameters
    ----------
    voters : array shape (a, n)
        Voter preferences; n-dimensional voter cardinal preferences for n issues. 
    candidates : array shape (b, n)
        Candidate preferences for b-candidates and n-dimensional issues. 
    tol : float, or array shaped (a,)
        Voter candidate tolerance distance. If cardinal preference exceed tol, 
        utility set to zero. 
    numwinners : int
        Number of winners for this election. 
    method : func
        Voting method function. Takes in argument array shaped (a, b)
        for (voters, candidates). 
    mtype : str
        Specify either a ranked or rated method. 
        
        - 'rank' -- Use candidate ranking method
        - 'score' -- Use candidate rating/scored method.
    
    Returns
    -------
    output : dict
        Dictionary of output keys
        
        - winners : integer array shaped (numwinners,)
            Candidate array indices of winners
        - ties : integer array shaped (numties,)
            Candidate array indices of ties, randomly included in winners
        -     

    """
    rating = estimate_rating(voters, candidates, tol=tol)
    ranks = irv.score2rank(rating)
    scores = np.round(rating * scoremax)
    if kwargs is None:
        kwargs = {}
        
    if mtype == 'rank':
        out1 = method(ranks, numwin=numwinners, **kwargs)
    elif mtype == 'score':
        out1 = method(scores, numwin=numwinners, **kwargs)
    else:
        raise ValueError('mtype not found')
        
        
    winners = out1[0]
    ties = out1[1]
    winners = handle_ties(winners, ties, numwinners)
        
    output = {}
        
    output['winners'] = winners
    output['ties'] = ties
    output['func output'] = out1[2:]
    output['candidate preferences'] = candidates
    output['voter preferences'] = voters
    output['scores'] = scores
    output['ranks'] = ranks
    output['rating'] = rating
    return output
    


def output_stats(output, bins=10):
    winners = output['winners']
    pref_candidates = output['candidate preferences']
    pref_voters = output['voter preferences']
    pref_winners = pref_candidates[winners]
    num_voters = len(pref_voters)
    num_candidates = len(pref_candidates)
    num_winners = len(winners)
    
    
    h_voters, h_edges = hist_norm(pref_voters, bins=bins)
    h_edges_c = np.copy(h_edges)
    h_edges_c[0] = pref_candidates.min()
    h_edges_c[-1] = pref_candidates.max()
    
    
    h_candidates, _ = hist_norm(pref_candidates, h_edges_c)
    h_winners, _ = hist_norm(pref_winners, h_edges)
    
    hist_error = np.sum(np.abs(h_winners - h_voters))
    avg_voter = np.mean(pref_voters)
    avg_winner = np.mean(pref_winners)
    avg_error = avg_voter - avg_winner
    
    std_voter = np.std(pref_voters)
    std_winner = np.std(pref_winners)
    std_error = std_voter - std_winner
    
    median_voter = np.median(pref_voters)
    median_winner = np.median(pref_winners)
    median_error = median_voter - median_winner
    
    return locals()


def hist_norm(x, bins=10):
    """
    Generate histogram data normalized by total population
    """
    h, edges = np.histogram(x, bins=bins)
    population = len(x)
    hnew = h / population
    return hnew, edges



def print_key(d, key):
    """Print to screen a key from dict generated by func `output_stats`
    
    Parameters
    ----------
    d : dict
        Dictionary output from function `output_stats`
    key : str
        Dictionary key for d

    """
    s = '%s = %.3f' % (key, d[key])
    print(s)
    
    
def plot_hist(output):
    """
    Plot histogram information from output from `simulate_election`
    """
    edges = output['h_edges']
    
    
    xedges = 0.5 * (edges[0:-1] + edges[1:])
    voters = output['h_voters'] 
    candidates = output['h_candidates']
    winners = output['h_winners']
    print(winners)
    plt.plot(xedges, voters, label='voters')
    plt.plot(xedges, candidates, 'o-', label='candidates')
    plt.plot(xedges, winners, 'o-', label='winners')
    plt.legend()


def generate_voter_preferences(numfactions, size, ndim=1, sepfactor=1, seed=None):
    """
    Create multi-peaked gaussian distributions of preferences
    
    Parameters
    ----------
    numvoters : int array of shape (a,), or int
        Number of voter preferences to generate. If list/array, generate 
        multiple gaussian voter preference peaks. 
    ndim : int, default=1
        Number of preference dimensions to consider
    sepfactor : float
        Scaling factor of typical distance of peak centers away from one another
    seed : None (default) or int
        Convert preference generation to become deterministic & pseudo-random
        for testing purposes. 
        
        - If None, use internal clock for seed generation
        - If int, use this seed to generate future random variables. 
        
    Returns
    -------
    out : array shaped (c, ndim)
        Voter preferences for ndim number of dimensions.
        
    Example
    -------
    
    Create a 2-dimensional preference spectrum, with 3 factions/preference 
    peaks with:
        - 200 voters in the 1st faction
        - 400 voters in the 2nd faction
        - 100 voters in the 3rd faction
    
    >>> p = generate_voter_preference((200, 400, 100), 2)
    
    Create a 1-dimensional preference spectrum with gaussian distribution
    for 500 voters.
    
    >>> p = generate_voter_preferences(500, 1)
    
    """
    
    rstate = np.random.RandomState(seed)
    
    
    numvoters = [rstate.randint(1, size) for i in range(numfactions)]
    new = []
    numvoters = np.atleast_1d(numvoters)
#    numleft = numvoters
    for pop_subset in numvoters:
        
#        if i == numpeaks - 1:
#            pop_subset = numleft
#        else:
#            pop_subset = np.random.randint(0, numleft)
        
        center = (rstate.rand(ndim) - 0.5) * sepfactor
        scale = rstate.rayleigh(size=ndim) / 2
        pi = rstate.normal(loc=center,
                              scale=scale, 
                              size=(pop_subset, ndim))
        new.append(pi)
#        numleft = numleft - pop_subset
    new = np.vstack(new)
    return new
        

class ElectionScenario1(object):
    
    def __init__(self, voters, candidates, method, mtype, 
                 numwinners=1, cnum=7, scoremax=5, tol=0.5, strategy='voter',
                 kwargs=None):
        self.voters = voters
        self.candidates = candidates
        
        self.method = method
        self.mtype = mtype
        self.kwargs = kwargs
        
        self.cnum = cnum
        self.scoremax = scoremax
        self.tol = tol
        self.strategy = strategy        
    
    def run(self):
        self.run_behavior()
        self.run_method()
        
        
    def run_behavior(self):
        cnum = 12
        scoremax = 5
        tol = .5
        strategy = 'voter'
        
#        ranks = behavior.voter_rankings(self.voters, self.candidates,
#                                        cnum=cnum,)
        rating = behavior.voter_scores_by_tolerance(self.voters, 
                                                    self.candidates,
                                                    tol=tol,
                                                    strategy=strategy)
        ranks = behavior.score2rank(rating)
        scores = np.round(rating * scoremax)
        self.ranks = ranks
        self.scores = scores
        self.ratings = rating
    
    
    def run_method(self):
        if self.mtype == 'rank':
            out1 = self.method(self.ranks,
                               numwin=self.numwinners, 
                               **self.kwargs)
        elif self.mtype == 'score':
            out1 = self.method(self.scores, 
                               numwin=self.numwinners, 
                               **self.kwargs)
        else:
            raise ValueError('mtype not found')       
            
        winners = out1[0]
        ties = out1[1]
        winners = handle_ties(winners, ties, numwinners)        
        self.winners = winners
        self.ties = ties
        self.func_output = out1[2:]     
        return out1
    
    
            
#
#class ElectionSim(object):
#    def __init__(self, method, mtype):
#        pass
#    
#    @staticmethod
#    def candidate_preference()

    
    
np.random.seed(None)
numvoters = 10000
numcandidates = 40
numwinners = 1

### Create array of voter prefere5nces
voters1 = np.random.normal(size=int(numvoters/2)) + 3
voters2 = np.random.normal(size=int(numvoters/2)) - 2
voters = np.append(voters1, voters2)
#voters = voters2
#np.random.seed(1)
candidates = np.random.rand(numcandidates) * 20 - 10 
tol = 1
#method = score.reweighted_range
#method = irv.IRV_STV
method = plurality.plurality
mtype = 'score'
#mtype = 'rank'
output = simulate_election(voters, candidates, tol, numwinners, 
                  method=method,
                  mtype=mtype,)
stat_output = output_stats(output)
plot_hist(stat_output)

print_key(stat_output, 'avg_error')
print_key(stat_output, 'std_error')
print_key(stat_output, 'median_error')
print_key(stat_output, 'hist_error')

#
#    output['winner_avg_preference'] = np.mean(candidates[winners])
#    output['winner_median_preference = np.median(candidates[winners])
#    winner_std_preference = np.std(candidates[winners])
#    voter_avg_preference = np.mean(voters)
#    voter_median_preference = np.median(voters)
#    voter_std_preference = np.std(voters)

    
        

#winners, ties, history = score.reweighted_range(scores, C_ratio=1, numwin=numwinners)
#winners, ties = plurality.plurality(scores, numwin=numwinners)
#winners = rcv.STV_calculator(ranks, winners=numwinners)

#
#h_voters, edges1 = np.histogram(voters, bins=20)
#h_candidates, edges2 = np.histogram(candidates, bins=20)
#h_winners, edges3 = np.histogram(candidates[winners], bins=20)
#
#
#
#print('voter avg preference = %.3f' % voter_avg_preference)
#print('voter median preference = %.3f' % voter_median_preference)
#print('voter std preference = %.3f' % voter_std_preference)
#print('winner avg preference = %.3f' % winner_avg_preference)
#print('winner median preference = %.3f' % winner_median_preference)
#print('winner std preference = %.3f' % winner_std_preference)
#print('')
#plt.figure()
#plt.plot(edges1[0:-1], h_voters / h_voters.max(), '.-', label='voters')
#plt.plot(edges2[0:-1], h_candidates / h_candidates.max(), '.-', label='candidates')
#plt.plot(edges3[0:-1], h_winners / h_winners.max(), 'o-', label='winners')
#plt.legend()