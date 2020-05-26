# -*- coding: utf-8 -*-
"""
Output controller & metrics to measure performance of an election
"""
import logging


import numpy as np

from votesim import utilities
from votesim.models import vcalcs

from votesim.votesystems.condcalcs import condorcet_check_one
from votesim.votesystems.tools import winner_check


logger = logging.getLogger(__name__)





#def interp_nearest(x, y):
#    x = np.array(x)
#    if x.shape[1] == 1:

class ElectionData(object):
    """Store temporary election results data here to make output calculations.
    Not meant to be used directly by user, created by ElectionStats.

    Attributes
    -----------
    voters :
        Voter preferences
    candidates :
        Candidate preferences
    winners :
        Election winners
    ballots :
        Ballot data used in election
    distances :
        Regret of voters for candidate; opposite of utility.

    **kwargs : any additional datas to set
    """
    def __init__(self,
                 voters=None,
                 weights=None,
                 order=1,
                 candidates=None,
                 winners=None,
                 distances=None,
                 ballots=None,
                 ties=None,
                 **kwargs):

        self.voters = voters
        self.weights = weights
        self.order = order

        self.candidates = candidates
        self.winners = winners
        self.distances = distances
        self.ballots = ballots
        self.ties = ties

        for k, v in kwargs.items():
            setattr(self, k, v)


    def set(self, **kwargs):
        """set kwargs to attributes"""
        for k, v in kwargs.items():
            setattr(self, k, v)






class ElectionStats(object):
    """
    ElectionStats collects election output data and re-routes that data
    towards various calculations and post-process variables.
    Output data tree



    """

    def __init__(self, voters=None, weights=-1, order=None, candidates=None,
                 winners=None, distances=None, ballots=None, **kwargs):
        self._electionData = ElectionData()

        self._output_categories = self._default_categories
        self._cache_voter = {}
        self._cache_candidate = {}
        self._cache_result = {}

        self.set_data(voters=voters, weights=weights, order=order,
                      candidates=candidates,
                      winners=winners,
                      distances=distances,
                      ballots=ballots,
                      **kwargs)
        return


    def set_data(self, voters=None, weights=-1, order=None, candidates=None,
                 winners=None, distances=None, ballots=None, ties=None,
                 **kwargs):
        """
        Set new election data, delete cached statistics
        """

        self._cache_result = {}

        if voters is not None:
            self.electionData.voters = voters
            self._cache_voter = {}
            self._cache_candidate = {}


        if weights != -1:
            self.electionData.weights = weights
            self._cache_voter = {}
            self._cache_candidate = {}


        if order is not None:
            self.electionData.order = order

        if candidates is not None:
            self.electionData.candidates = candidates
            self._cache_candidate = {}

        if winners is not None:
            self.electionData.winners = winners

        if ballots is not None:
            self.electionData.ballots = ballots

        if ties is not None:
            self.electionData.ties = ties

        ### Calculate voter distances
        calculate = False
        if distances is None:
            if ((self.electionData.candidates is not None) and
                (self.electionData.voters is not None)):
                calculate = True
        else:
            self.electionData.distances = distances

        if calculate:
            self.electionData.distances = vcalcs.voter_distances(

                voters=self.electionData.voters,
                candidates=self.electionData.candidates,
                weights=self.electionData.weights,
                order=self.electionData.order,

                )
        self.electionData.set(**kwargs)
        return


    _default_categories = [
        'voter',
        'candidate',
        'winner',
        'winner_categories',
        'ballot'
        ]


    def set_categories(self, names, fulloutput=False):
        """
        Set output categories to output.

        Parameters
        ----------
        names : list of str
            Output category names.
        fulloutput : bool, optional
            If True output all avaialable ouputs. The default is False.

        Returns
        -------
        None.

        """
        if fulloutput == True:
            names = self.get_categories()

        self._output_categories = names
        return


    def get_categories(self):
        """Retrieve available output categories"""
        return self._default_categories


    def add_output(self, name, output, cache='_cache_result'):
        """Add an output object. This output's base class must be
        :class:`~votesim.metrics.BaseStats`.

        Parameters
        ----------
        name : str
            Name of output
        output : subtype of :class:`~votesim.metrics.BaseStats`
            User defined output. Define this output by creating a class
            inherited from :class:`~votesim.metrics.BaseStats`
        cache : str
            Name of output cache to store results. This determines when
            output is retained and when it is deleted and regenerated
            during election model creation. The options are

            - '_cache_voter' - Clear cache when voter data changes (least aggressive)
            - '_cache_candidate' - Clear cache when candidate data changes
            - '_cache_result' - Clear cache after every election (most aggressive)

        Returns
        -------
        None.

        """
        if hasattr(self, name):
            s = 'Name %s for output already taken. Use another' % name
            raise ValueError(s)

        # Set cache decorator. The default clears cache every new election.
        output = utilities.lazy_property2(cache)(output)

        setattr(self, name, output)
        #self._default_categories.append(name)
        self._output_categories.append(name)
        return


    def get_dict(self):
        """Retrieve desired category key and values and return dict of dict."""
        d = {}
        for key in self._output_categories:
            stat = getattr(self, key)
            di = stat._dict
            d[key] = di
        return d


    def get_docs(self):
        """Retrieve all available statistic descriptions as dict"""
        d = {}
        for key in self._output_categories:
            stat = getattr(self, key)
            di = stat._docs
            d[key] = di
        return d



    @property
    def electionData(self):
        """Temporary election data used for output calculations"""
        return self._electionData


    @utilities.lazy_property2('_cache_voter')
    def voter(self):
        """See :class:`~votesim.metrics.VoterStats`"""
        return VoterStats(self)


    @utilities.lazy_property2('_cache_candidate')
    def candidate(self):
        """See :class:`~votesim.metrics.CandidateStats`"""
        return CandidateStats(self)


    @utilities.lazy_property2('_cache_result')
    def winner(self):
        """See :class:`~votesim.metrics.WinnerStats`"""
        return WinnerStats(self)


    @utilities.lazy_property2('_cache_result')
    def winner_categories(self):
        """See :class:`~votesim.metrics.WinnerCategories`"""
        return WinnerCategories(self)


    @utilities.lazy_property2('_cache_result')
    def ballot(self):
        """See :class:`~votesim.metrics.BallotStats`"""
        return BallotStats(self)


class BaseStats(object):
    """Base inheritance class for Stats objects. Use this to create
    new Statistic Output objects.

    All attributse that do not start with underscore '_' will be used as
    output variables to be stored.

    Parameters
    ------------
    electionStats : `ElectionStats`
        ElectionStats parent object

    Attributes
    -------------
    _electionStats : ElectionStats
        Top-level output object
    _electionData : ElectionData
        Temporary election data used for making calculations


    Example
    -------
    Create your new output object


    >>> import numpy as np
    >>>
    >>> class MyStats(BaseStats):
    >>>     @votesim.utilities.lazy_property
    >>>     def stat1(self):
    >>>         v = self._electionData.voters
    >>>         return np.mean(v, axis=0)



    """
    def __init__(self, electionStats: ElectionStats):
        self._electionStats = electionStats
        self._electionData = electionStats.electionData
        self._reinit()
        return

    def _reinit(self):
        """Define custom initialization routines here"""
        return


    @utilities.lazy_property
    def _keys(self):
        """Retrieve output keys as list"""
        a = dir(self)
        new = []
        for name in a:
            if name.startswith('_'):
                pass
            else:
                new.append(name)
        return new


    @property
    def _dict(self):
        """Retrieve all statistics output and return as `dict`"""
        keys = self._keys
        return {k: getattr(self, k) for k in keys}


    def _docs(self):
        """Retrieve all descriptions of outputs and return as dict"""
        clss = type(self)
        new = {}
        for key, attrname in self._dict.items():
            doc = getattr(clss, attrname).__doc__

            # Get rid of newlines in docstring
            doc = doc.replace('\n', ' ')

            # Get rid of too much whitespace
            doc = ' '.join(doc.split())
            new[key] = doc
        return new



class VoterStats(BaseStats):
    """Voter population statistics

    """


    def _reinit(self):
        ed = self._electionData
        self._voters = ed.voters
        self._weights = ed.weights
        self._order = ed.order
        return


    @utilities.lazy_property
    def regret_mean(self):
        """Regret of voters if winner is located at preference mean"""
        return mean_regret(self._voters, self._weights, order=self._order)


    @utilities.lazy_property
    def regret_median(self):
        """Regret of voters if winner is located at preference median"""
        return median_regret(self._voters, self._weights)


    @utilities.lazy_property
    def regret_random_avg(self):
        """Average regret of voters if winner is randomly selected from voter
        population"""

        r = voter_regrets(self._voters,
                          self._weights,
                          maxsize=5000,
                          order=self._order,
                          seed=0)
        return np.mean(r)


    @utilities.lazy_property
    def pref_mean(self):
        """"array shape (n) : Preference mean of voters for n preference dimensions"""
        return np.mean(self._voters, axis=0)


    @utilities.lazy_property
    def pref_median(self):
        """array shape (n) : Preference median of voters for n preference dimensions"""
        return np.median(self._voters, axis=0)


    @utilities.lazy_property
    def pref_std(self):
        """array shape (n) : Preference standard deviation of voters for
        n preference dimensions"""
        return np.std(self._voters, axis=0)


class CandidateStats(BaseStats):
    """Candidate statistics dependent on
    :class:`~votesim.metrics.ElectionStats.voters`"""

    def _reinit(self):
        ed = self._electionData
        self._distances = ed.distances
        return


    @utilities.lazy_property
    def pref(self):
        """array shape (a, b) : Candidate preference coordinates"""
        return self._electionStats.electionData.candidates


    @utilities.lazy_property
    def regrets(self):
        """array shape (c) : voter regret for each candidate"""
        distances = self._distances
        return np.mean(distances, axis=0)


    @utilities.lazy_property
    def _regret_best(self):
        """Retrieve best regrests and corresponding winner indices"""
        regrets = self.regrets

        ii = np.argsort(regrets)
        ii_ideal = ii[0]
        ri = np.mean(regrets[ii_ideal])
        return ri, ii_ideal


    @property
    def regret_best(self):
        """Best possible regret for the best candidate in election"""
        return self._regret_best[0]


    @utilities.lazy_property
    def regret_avg(self):
        """Average regret if a random candidate became winner"""
        return np.mean(self.regrets)


    @property
    def winner_utility(self):
        """Best utility candidate in election"""
        return self._regret_best[1]


    @utilities.lazy_property
    def winner_condorcet(self):
        """Condorcet winner of election, return -1 if no condorcet winner found."""
        distances = self._distances
        return condorcet_check_one(scores=-distances)


    @utilities.lazy_property
    def _winner_plurality_calcs(self):
        """Plurality winner of election; return -1 if tie found

        Returns
        ---------
        winner : int
            Candidate index of plurality winner
        votes : int
            Number of votes cast for plurality winner
        counts : array shape (a,)
            Vote counts for all candidates
        """
        distances = self._distances
        ii = np.argmin(distances, axis=1)
        ulocs, ucounts = np.unique(ii, return_counts=True)

        counts = np.zeros(distances.shape[1], dtype=int)
        counts[ulocs] = ucounts
        votes = np.max(counts)

        winner, ties = winner_check(counts, numwin=1)
        if len(ties) > 1:
            winner = -1
        else:
            winner = winner[0]
        return winner, votes, counts


    @property
    def winner_plurality(self):
        """Plurality winning candidate of election"""
        return self._winner_plurality_calcs[0]


    @utilities.lazy_property
    def winner_majority(self):
        """Majority winner of election; return -1 if no majority found"""

        winner, votes, counts = self._winner_plurality_calcs
        vnum = len(self._distances)

        if votes > vnum/2.:
            return winner
        else:
            return -1


    @utilities.lazy_property
    def plurality_ratio(self):
        """float : Ratio of plurality winning votes to total votes.
        This metric attempts to measure to competitiveness of an election."""
        votes = self._winner_plurality_calcs[1]
        vnum = len(self._distances)
        return float(votes) / vnum


    @utilities.lazy_property
    def utility_ratio(self):
        """Utility ratio of the best candidate compared to average candidate,
        normalized by the utility range from random to ideal candidate. This
        metric attempts to measure if there's a clear stand-out winner in
        the election."""

        v_median = self._electionStats.voter.regret_median
        #v_rand = self._electionStats.voter.regret_random_avg
        v_best = self.regret_best
        v_avg = self.regret_avg
        return (v_avg - v_best) / (v_avg - v_median)




class WinnerStats(BaseStats):
    """Winner output statistics"""

    def _reinit(self):
        self._candidate_regrets = self._electionStats.candidate.regrets
        self._winners = self._electionStats.electionData.winners
        return


    @utilities.lazy_property
    def regret(self):
        """overall satisfaction of all winners for all voters."""
        candidate_regrets = self._candidate_regrets
        ii = self._winners
        winner_regrets = candidate_regrets[ii]
        return np.mean(winner_regrets)


    @utilities.lazy_property
    def regret_efficiency_candidate(self):
        """Voter satisfaction efficiency, compared to random candidate"""

        random = self._electionStats.candidate.regret_avg
        best = self._electionStats.candidate.regret_best

        U = self.regret
        R = random
        B = best
        vse = (U - R) / (B - R)
        return vse


    @utilities.lazy_property
    def regret_efficiency_voter(self):
        """voter satisfaction efficiency equation normalizing to voter 
        population regret of an ideal winner vs a random voter.
        """

        v_random = self._electionStats.voter.regret_random_avg
        v_median = self._electionStats.voter.regret_median
        best = self._electionStats.candidate.regret_best

        U = self.regret
        R2 = v_random
        R1 = v_median
        B = best

        return 1.0 - abs(U - B) / (R2 - R1)


    @utilities.lazy_property
    def regret_normed(self):
        """Voter regret normalized to ideal"""
        U = self.regret
        R = self._electionStats.voter.regret_median
        return U / R - 1


    @property
    def winners(self):
        """array of int : Index location of winners"""
        return self._electionData.winners

    @property
    def ties(self):
        """array of int : Index location of ties"""
        return self._electionData.ties


class WinnerCategories(BaseStats):
    """Determine whether majority, condorcet, or utility winner was elected"""


    def _reinit(self):
        self._winners = self._electionStats.electionData.winners
        return


    @utilities.lazy_property
    def is_condorcet(self):
        """bool : check whether condorcet winner was elected."""
        ii = self._electionStats.candidate.winner_condorcet
        if self._winners[0] == ii:
            return True
        return False


    @utilities.lazy_property
    def is_majority(self):
        """bool : check if majority winner was elected."""
        ii = self._electionStats.candidate.winner_majority
        if self._winners[0] == ii:
            return True
        return False


    @utilities.lazy_property
    def is_utility(self):
        """bool : check if utility winner was elected"""
        ii = self._electionStats.candidate.winner_utility
        if self._winners[0] == ii:
            return True
        return False




class BallotStats(BaseStats):
    """Ballot marking statistics"""

    def _reinit(self):
        ed = self._electionData
        self._ballots = ed.ballots
        return


    @utilities.lazy_property2('_cache_ballot')
    def _ballot_stats(self) -> dict:
        ballots = np.atleast_2d(self._ballots)
        ballot_num, candidate_num = ballots.shape

        # Get number of candidates marked for each ballot
        marked_array = np.sum(ballots > 0, axis=1)

        # Get ballots where bullet voting happened
        bullet_num = np.sum(marked_array == 1)
        bullet_ratio = bullet_num / ballot_num


        #Get ballots where all but one candidate is marked
        full_num = np.sum(marked_array >= (candidate_num - 1))
        full_ratio = full_num / ballot_num

        marked_num = np.sum(marked_array)
        marked_avg = np.mean(marked_array)
        marked_std = np.std(marked_array)

        d = {}
        d['ballot.bullet.num'] = bullet_num
        d['ballot.bullet.ratio'] = bullet_ratio
        d['ballot.full.num'] = full_num
        d['ballot.full.ratio'] = full_ratio
        d['ballot.marked.num'] = marked_num
        d['ballot.marked.avg'] = marked_avg
        d['ballot.marked.std'] = marked_std
        return d


    @property
    def bullet_num(self):
        """Number of ballots where voters only bullet voted for 1 candidate"""
        return self._ballot_stats['ballot.bullet.num']


    @property
    def bullet_ratio(self):
        """Ratio of ballots where voters only bullet voted for 1 candidate"""
        return self._ballot_stats['ballot.bullet.ratio']


    @property
    def full_num(self):
        """Number of ballots where all but one candidate is marked"""
        return self._ballot_stats['ballot.bullet.ratio']


    @property
    def full_ratio(self):
        """Ratio of ballots where all but one candidate is marked"""
        return self._ballot_stats['ballot.bullet.ratio']


    @property
    def marked_num(self):
        """Total number of marked candidates for all ballots"""
        return self._ballot_stats['ballot.marked.num']


    @property
    def marked_avg(self):
        """Average number of marked candidates per ballot"""
        return self._ballot_stats['ballot.marked.avg']


    @property
    def marked_std(self):
        """Std deviation of marked candidates per ballot"""
        return self._ballot_stats['ballot.marked.std']





class PrRegret(BaseStats):
    """Metrics for proportional representation"""
    def _reinit(self):
        ed = self._electionData
        self._distances = ed.distances
        self._num_voters, self._num_candidates = self._distances.shape
        self._num_winners = len(self._electionData.winners)
        return


    @utilities.decorators.lazy_property
    def _nearest_winners(self):
        """array shape (a,)
            index locations of the nearest winners for each voter,
            for `a` total voters
        """
        return np.argmin(self._distances, axis=1)


    @utilities.decorators.lazy_property
    def _nearest_winner_distances(self):
        """array shaped (a,)
            Preference distances of nearest winner for `a` voters.
        """
        ii = np.arange(self._num_voters)
        jj = self._nearest_winners

        return self._distances[ii, jj]



    @utilities.decorators.lazy_property
    def avg_regret(self):
        """float
            Average voter regret for his nearest winner
        """
        distances = self._nearest_winner_distances
        num_voters = self._num_voters
        num_winners = self._num_winners

        regret = np.sum(distances) / num_voters
        regret = regret * num_winners
        return regret


    @utilities.decorators.lazy_property
    def winners_regret(self):
        """array shaped (b,):
            Avg voter regrets for each winner
        """
        num_voters = self._num_voters
        num_winners = self._num_winners

        sregrets = []
        for ii in range(num_winners):
            index = (ii == self._nearest_winners)
            distances = self._nearest_winner_distances[index]
            regret = np.sum(distances)
            sregrets.append(regret)

        sregrets = np.array(sregrets) / num_voters * num_winners
        return sregrets


    @utilities.decorators.lazy_property
    def winners_regret_std(self):
        """float:
            Standard deviation of nearest regrets for each winner. An ideal
            proportional system ought to have low std deviation.
        """
        return np.std(self.winners_regret)


    @utilities.decorators.lazy_property
    def std_num_voters_per_winner(self):
        """float:
            Standard deviation of number of nearest voters for each winner
        """
        num_voters = self._num_voters
        num_winners = self._num_winners

        wcounts = []
        for ii in range(num_winners):
            wcount = np.sum(ii == self._nearest_winners)
            wcounts.append(wcount)

        voters_per_winner = num_voters / num_winners
        std = np.std(wcounts) / voters_per_winner
        return std


def candidate_regrets(voters, candidates, weights=None, order=1):
    """Calculate the voter regret for each candidate or winner.

    Parameters
    -----------
    voters : array (a, n)
        Voter preferences; n-dimensional voter cardinal preferences for n issues.
    candidates : array (b, n)
        Candidate preferences for `b` candidates and `n`-dimensional issues.


    Returns
    -------
    out : array (b,)
        Average preference distance of voters from each candidate numbering `b`.
    """

    voters = np.atleast_2d(voters)
    candidates = np.atleast_2d(candidates)
    num_voters = len(voters)

    # distance shape (a, b) for `a` num voters, `b` num candidates.
    distances = vcalcs.voter_distances(voters,
                                       candidates,
                                       weights=weights,
                                       order=order)

    avg_distances = np.sum(distances, axis=0) / num_voters
    return avg_distances


def voter_regrets(voters, weights=None, order=1, pnum=10, maxsize=5000, seed=None):
    """Calculate the voter regrets for each other if voters became a candidate

    Parameters
    ----------
    voters : array shape (a, n)
        Voter preferences; `a` number of voters, cardinal preferences for `n` issues.
    weights : None or array shape (a, n)
        Dimensional weightings of each voter for each dimension.
        Only relevant if n > 1
    order : int
        Order of norm

        * 1 = taxi-cab norm; preferences for each issue add up
        * 2 = euclidean norm; take the sqrt of squares.
    pnum : int
        Number of voters to calculate distances for at-a-time, for memory issues
    maxsize: int
        For large populations this calculation is expensive. Use this to sample
        a subset of the voter population. Default 5000.
        Set to None to use all voters.

    Returns
    -------
    out : array shape (c,)
        Voter regrets for each voter as a proposed candidate.

        - c = a if maxsize <= number voters or maxsize==None
        - c = maxsize otherwise for sampled voters.

    """

    cnum = len(voters)
    if maxsize is not None:
        if cnum > maxsize:
            rs = np.random.RandomState(seed)
            ii = rs.choice(cnum, size=maxsize, replace=False)
            voters = voters[ii]

    numbers = np.arange(0, cnum + pnum, pnum)
    lb_nums = numbers[0:-1]
    ub_nums = numbers[1:]


    dlist = []
    for lb, ub in zip(lb_nums, ub_nums):
        candidatesi = voters[lb : ub]
        try:
            d = candidate_regrets(voters, candidatesi, weights=weights, order=order)
            dlist.append(d)
        except MemoryError:
            return voter_regrets(voters, weights, order, pnum=1)

    return np.concatenate(dlist)



def consensus_regret(voters, winners, _distances=None):
    """
    Measure overall average satisfaction of all winners for all voters.

    Parameters
    -----------
    voters : array, shape (a, n)
        Voter preferences; n-dimensional voter cardinal preferences for n issues.
    winners : array, shape (b, n)
        Winner preferences for `b` winners and `n`-dimensional issues.

    Returns
    -------
    regret : float
        Consensus voter regret metric
    """
    num_winners = len(winners)
    if _distances is not None:
        distances = _distances
    else:
        distances = candidate_regrets(voters, winners)
    regret = np.sum(distances) / num_winners
    return regret


def mean_regret(voters, weights=None, order=1):
    """
    Measure overall regret of voters if a candidate located at the centroid
    was elected.

    Parameters
    -----------
    voters : array, shape (a, n)
        Voter preferences; n-dimensional voter cardinal preferences for n issues.
    weights : array, shape (a, n)
        Voter preference weights for each preference. (ie, voters care
        more about some preferences than others).

    """
    num = len(voters)
    center = np.mean(voters, axis=0)

    if weights is None:
        diff = voters - center
    else:
        diff = (voters - center) * weights
    dist = np.sum(np.linalg.norm(diff, axis=1, ord=order)) / num
    return dist


def median_regret(voters, weights=None, order=1):
    num = len(voters)
    center = np.median(voters, axis=0)
    if weights is None:
        diff = voters - center
    else:
        diff = (voters - center) * weights
    dist = np.sum(np.linalg.norm(diff, axis=1, ord=order)) / num
    return dist


def regret_std(voters, meanvoter=None, weights=None, order=1):
    if meanvoter is None:
        v_mean = np.mean(voters, axis=0)
    else:
        v_mean = meanvoter
    v_dist = vcalcs.voter_distances(voters,
                                    v_mean[None, :],
                                    weights=weights,
                                    order=order)
    std = np.std(v_dist)
    return std


#
#def _ballot_stats(self, election):
#
#    scores = election.scores
#    ranks = election.ranks
#    ratings = election.ratings
#
#    num_scored = np.sum(scores > 0, axis=1)
#    num_ranked = np.sum(ranks > 0, axis=1)
#    num_rated = np.sum(ratings > 0, axis=1)
#
#    self.avg_num_rated = np.average(num_rated)
#    self.avg_num_scored = np.average(num_scored)
#    self.avg_num_ranked = np.average(num_ranked)
#    self.std_num_scored = np.std(num_scored)
#    self.std_num_ranked = np.std(num_ranked)
#
#


class __ElectionStats_OLD(object):
    """Calculate and store various regret metrics

    Parameters
    ----------
    voters : array, shape (a, n)
        Voter preferences; n-dimensional voter cardinal preferences for n issues.

    candidates : array (b, n)
        Candidate preferences for `b` candidates and `n`-dimensional issues.

    winners : array, shape (b, n)
        Winner preferences for `b` winners and `n`-dimensional issues.

    ballots : array, shape (a, b)
        Submitted ballots, rows as voters & columns as candidates

        - Zero data ballots are minimum score, unranked, or not chosen candidates.
    order : int
        Order of regret distance calculations (default 1)

    Usage
    ------
    To access all metrics, use

    >>> self.get_dict()


    To retrieve descriptions for all matrics, use

    >>> self.get_docs()


    """
    def __init__(self, voters=None, weights=None,
                 candidates=None, winners=None, ballots=None, order=1):
        self.stats = {}
        self._voters = voters
        self._weights = weights
        self._candidates = candidates
        self._winners = winners
        self._ballots = ballots
        self._order = order

        self._cache_voter = {}
        self._cache_candidate = {}
        self._cache_result = {}
        self._distances = None

#        self.run(voters=voters,
#                 weights=weights,
#                 candidates=candidates,
#                 winners=winners,
#                 ballots=ballots)
        return


    def set(self,
            voters=None,
            weights=-1,
            candidates=None,
            winners=None,
            ballots=None):
        """Set voters, weights, candidates or winners to recalculate"""

        if voters is not None:
            self._voters = voters
            self._cache_voter = {}
            self._cache_candidate = {}
            self._cache_result = {}


        if weights != -1:
            self._weights = weights
            self._cache_voter = {}
            self._cache_candidate = {}
            self._cache_result = {}

        if candidates is not None:
            self._candidates = candidates
            self._cache_candidate = {}
            self._cache_result = {}

        if winners is not None:
            self._winners = winners
            self._cache_result = {}

        if ballots is not None:
            self._ballots = ballots


        self._distances = vcalcs.voter_distances(voters=self._voters,
                                                 candidates=self._candidates,
                                                 weights=self._weights,
                                                 order=1
                                                 )





#    def run(self, voters=None, weights=None,
#            candidates=None, winners=None, ballots=None):
#
#        d = self.stats.copy()
#        if voters is not None:
#            stats1 = self._voter_stats(voters, weights)
#            d.update(stats1)
#            self.stats = d
#
#        if (candidates is not None) and (winners is not None):
#            stats2 = self._result_stats(voters, candidates, winners, weights)
#            d.update(stats2)
#
#        if ballots is not None:
#            stats3 = self._ballot_stats(ballots)
#            d.update(stats3)
#
#        self.stats = d
#        return



    def _get_category_keys(self, category):
        """Divide metrics into categories defined by an attribute prefix"""
        a = dir(self)
        prefix = category + '_'

        new = {}
        for name in a:
            if name.startswith(prefix):
                newkey = name.replace(prefix, category + '.')
                new[newkey] = name
        return new


    @property
    def _keys_voter(self):
        """Retrieve voter metrics' attribute names"""
        category = 'voter'
        return self._get_category_keys(category)

    @property
    def _keys_candidate(self):
        """Retrieve candidate metrics' attribute names"""
        category = 'candidate'
        return self._get_category_keys(category)

    @property
    def _keys_regret(self):
        """Retrieve regret metrics' attribute names"""
        category = 'regret'
        return self._get_category_keys(category)

    @property
    def _keys_winner(self):
        """Retrieve winner metrics' attribute names"""
        category = 'winner'
        return self._get_category_keys(category)

    @property
    def _keys_ballot(self):
        """Retrieve ballot metrics' attribute names"""
        category = 'ballot'
        return self._get_category_keys(category)


    @utilities.lazy_property
    def _keydict(self):
        """Retrieve dict keynames that retrieve attribute data"""
        new = {}
        new.update(self._keys_voter)
        new.update(self._keys_candidate)
        new.update(self._keys_regret)
        new.update(self._keys_winner)
        new.update(self._keys_ballot)
        return new


    def get_keys(self):
        """Get a list of all available statistics"""
        return list(self._keydict.keys())


    def get_dict(self):
        """Retrieve all available statistics"""
        new = {}
        for key, attrname in self._keydict.items():
            try:
                new[key] = getattr(self, attrname)
            except RuntimeError:
                pass
        return new


    def get_docs(self):
        """Retrieve all available statistic descriptions as dict"""
        clss = type(self)
        new = {}
        for key, attrname in self._keydict.items():
            doc = getattr(clss, attrname).__doc__
            doc = doc.replace('\n', ' ')
            doc = ' '.join(doc.split())
            new[key] = doc
        return new


    @property
    def voters(self):
        if self._voters is None:
            raise RuntimeError('Voters are not yet defined in Metrics')
        return self._voters

    @property
    def weights(self):
        return self._weights


    @property
    def candidates(self):
        if self._candidates is None:
            raise RuntimeError('Candidates are not yet defined in Metrics')
        return self._candidates


    @property
    def winners(self):
        if self._winners is None:
            raise RuntimeError('Winners are not yet defined in Metrics')
        return self._winners


    @property
    def ballots(self):
        if self._ballots is None:
            raise RuntimeError('Ballots are not yet defined in Metrics')
        return self._ballots


    ### Metrics


    @utilities.lazy_property2('_cache_voter')
    def voter_regret_mean(self):
        """Regret of voters if winner is located at preference mean"""
        return mean_regret(self.voters, self.weights, order=self._order)


    @utilities.lazy_property2('_cache_voter')
    def voter_regret_median(self):
        """Regret of voters if winner is located at preference median"""
        return median_regret(self.voters, self.weights)


    @utilities.lazy_property2('_cache_voter')
    def voter_regret_random_avg(self):
        """Average regret of voters if winner is randomly selected from voter
        population"""

        r = voter_regrets(self.voters,
                          self.weights,
                          maxsize=5000,
                          order=self._order,
                          seed=0)
        return np.mean(r)


    @utilities.lazy_property2('_cache_voter')
    def voter_mean(self):
        """"array shape (n) : Preference mean of voters for n preference dimensions"""
        return np.mean(self.voters, axis=0)


    @utilities.lazy_property2('_cache_voter')
    def voter_median(self):
        """array shape (n) : Preference median of voters for n preference dimensions"""
        return np.median(self.voters, axis=0)


    @utilities.lazy_property2('_cache_voter')
    def voter_std(self):
        """array shape (n) : Preference standard deviation of voters for
        n preference dimensions"""
        return np.std(self.voters, axis=0)


    @utilities.lazy_property2('_cache_voter')
    def voter_regret_std(self):
        """Standard deviation of regret """
        meanvoter = self.voter_mean
        return regret_std(self.voters,
                          meanvoter=meanvoter,
                          weights=self.weights,
                          order=self._order)


    @utilities.lazy_property2('_cache_candidate')
    def candidate_regrets(self):
        """array shape (c) : voter regret for each candidate"""
        return candidate_regrets(self.voters,
                                 self.candidates,
                                 order=self._order)


    # @utilities.lazy_property2('_cache_result')
    # def _PR_regret(self):
    #     pr = PrRegret(self.voters, self.winners, self.weights)
    #     regret = pr.regret
    #     std_regret = pr.std_regret
    #     return regret, std_regret


    # @property
    # def regret_PR(self):
    #     """Multi-winner average regret for Proportional Representation.
    #     Average voter regret for his nearest winner"""
    #     return self._PR_regret[0]


    # @property
    # def regret_PR_std(self):
    #     """Standard deviation of nearest regrets for each winner. An ideal
    #     proportional system ought to have low std deviation"""
    #     return self._PR_regret[1]


    @utilities.lazy_property2('_cache_result')
    def winner_num(self):
        """Number of winners for this election"""
        return len(self.winners)


    @property
    def winner_all(self):
        """All winners of election"""
        return self.winners


    @utilities.lazy_property2('_cache_result')
    def regret_consensus(self):
        """overall satisfaction of all winners for all voters."""
        candidate_regrets = self.candidate_regrets
        ii = self.winners

        winner_pref = self.candidates[ii]

        rc = consensus_regret(self.voters,
                              winner_pref,
                              _distances=candidate_regrets[ii])
        return rc


    @utilities.lazy_property2('_cache_candidate')
    def _regret_best(self):
        """Retrieve best regrests and corresponding winner indices"""
        candidate_regrets = self.candidate_regrets
        winner_num = self.winner_num

        ii = np.argsort(candidate_regrets)
        ii_ideal = ii[0 : winner_num]
        ri = np.mean(candidate_regrets[ii_ideal])
        return ri, ii_ideal


    @property
    def regret_best(self):
        """Best possible regret for the best candidate in election"""
        return self._regret_best[0]


    @property
    def candidate_best(self):
        """Best possible candidate (in terms of regret) in election"""
        return self._regret_best[1]


    @utilities.lazy_property2('_cache_candidate')
    def candidate_regret_random(self):
        """Average regret if a random candidate became winner"""
        return np.mean(self.candidate_regrets)


    @property
    def candidate_preference(self):
        """Preference locations of candidates"""
        return self.candidates


    @property
    def regret_efficiency_candidate(self):
        """Voter satisfaction efficiency, compared to random candidate"""
        U = self.regret_consensus
        R = self.candidate_regret_random
        B = self.regret_best
        vse = (U - R) / (B - R)
        return vse


    @property
    def regret_efficiency_voter(self):
        """My updated satisfaction efficiency equation normalizing to voter population
        rather than candidate population"""
        U = self.regret_consensus
        R2 = self.voter_regret_random_avg
        R1 = self.voter_regret_median
        B = self.regret_best

        return 1.0 - abs(U - B) / (R2 - R1)


    @property
    def regret_normed(self):
        """Voter regret normalized to ideal"""
        U = self.regret_consensus
        R = self.voter_regret_median
        return U / R - 1





    @utilities.lazy_property2('_cache_ballot')
    def _ballot_stats(self):
        ballots = np.atleast_2d(self.ballots)
        ballot_num, candidate_num = ballots.shape

        # Get number of candidates marked for each ballot
        marked_array = np.sum(ballots > 0, axis=1)

        # Get ballots where bullet voting happened
        bullet_num = np.sum(marked_array == 1)
        bullet_ratio = bullet_num / ballot_num


        #Get ballots where all but one candidate is marked
        full_num = np.sum(marked_array >= (candidate_num - 1))
        full_ratio = full_num / ballot_num

        marked_num = np.sum(marked_array)
        marked_avg = np.mean(marked_array)
        marked_std = np.std(marked_array)


        d = {}
        d['ballot.bullet.num'] = bullet_num
        d['ballot.bullet.ratio'] = bullet_ratio
        d['ballot.full.num'] = full_num
        d['ballot.full.ratio'] = full_ratio
        d['ballot.marked.num'] = marked_num
        d['ballot.marked.avg'] = marked_avg
        d['ballot.marked.std'] = marked_std
        return d




    @property
    def ballot_bullet_num(self):
        """Number of ballots where voters only bullet voted for 1 candidate"""
        return self._ballot_stats['ballot.bullet.num']


    @property
    def ballot_bullet_ratio(self):
        """Ratio of ballots where voters only bullet voted for 1 candidate"""
        return self._ballot_stats['ballot.bullet.ratio']


    @property
    def ballot_full_num(self):
        """Number of ballots where all but one candidate is marked"""
        return self._ballot_stats['ballot.bullet.ratio']


    @property
    def ballot_full_ratio(self):
        """Ratio of ballots where all but one candidate is marked"""
        return self._ballot_stats['ballot.bullet.ratio']


    @property
    def ballot_marked_num(self):
        """Total number of marked candidates for all ballots"""
        return self._ballot_stats['ballot.marked.num']


    @property
    def ballot_marked_avg(self):
        """Average number of marked candidates per ballot"""
        return self._ballot_stats['ballot.marked.avg']


    @property
    def ballot_marked_std(self):
        """Std deviation of marked candidates per ballot"""
        return self._ballot_stats['ballot.marked.std']


#    def _result_stats(self, voters, candidates, winners, weights):
#        regret_candidates = candidate_regrets(voters, candidates)
#        rr = np.mean(regret_candidates)
#
#        ### Average satisfaction of voter to closest winner
#
#        pr = PrRegret(voters, winners, weights)
#
#        ### Overall satisfaction of all voters for all winners
#        winner_pref = candidates[winners]
#        winner_num = len(winners)
#        ii = winners
#        rc = consensus_regret(voters,
#                             winner_pref,
#                             _distances=regret_candidates[ii])
#
#        ### Minimum possible consensus regret for this election
#
#
#        regret_best, candidate_best = self._regret_best(regret_candidates, winner_num)
#        vse = self._vse(rc, rr, regret_best)
#
#        rvm = self.stats['voter.regret.median']
#        regret_median_acc = self._regret_median_accuracy(rc, rvm)
#
#        d = {}
#        d['regret.candidates'] = regret_candidates
#        d['regret.PR'] = pr.regret
#        d['regret.PR_std'] = pr.std_regret
#
#        d['regret.consensus'] = rc
#        d['regret.best'] = regret_best
#        d['regret.random'] = rr
#        d['regret.vse'] = vse
#        d['regret.vsp'] =  self._satisfaction_population(rc, regret_best, rvm)
#        d['regret.median_accuracy'] = regret_median_acc
#        d['winners.num'] = winner_num
#        d['winners'] = winners
#        d['candidates.preference'] = candidates
#        d['candidates.best'] = candidate_best
#        return d



#
#if __name__ == '__main__':
#    rs = np.random.RandomState(None)
#    win_num = np.arange(5, 100, 1)
#    regrets = []
#    ndim = 1
#    for w in win_num:
#        voters = rs.rand(5000, ndim) * 10
#        winners = rs.rand(w, ndim) * 10
#        r = PR_regret(voters, winners)
#        regrets.append(r)
#
#
#    import matplotlib.pyplot as plt
#    plt.plot(win_num, regrets)
#    plt.ylim(0, None)
#
#



