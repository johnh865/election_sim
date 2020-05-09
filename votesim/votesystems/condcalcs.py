"""
Functions for calculating Condorcet-related voting methods
"""
import logging
import numpy as np

from votesim.votesystems.tools import RCV_reorder, winner_check
from votesim.utilities.decorators import lazy_property
logger = logging.getLogger(__name__)





def pairwise_rank_matrix(ranks):
    """
    Calculate total votes for a candidate against another candidate given
    ranked voter data.

    Parameters
    ----------
    ranks : array shaped (a, b)
        Election voter rankings, from [1 to b].
        Data composed of candidate rankings, with

           - Voters as the rows
           - Candidates as the columns.

        Use 0 to specify unranked (and therefore not to be counted) ballots.

        - a : number of voters dimension. Voters assign ranks for each candidate.
        - b : number of candidates. A score is assigned for each candidate
              from 0 to b-1.

    Returns
    --------
    out : array shaped (b, b,)
        Vote matrix V[i,j]

        - Total votes for candidate i against candidate j

    """
    data = np.atleast_2d(ranks).copy()
    data = RCV_reorder(data)
    cnum = data.shape[1]

    # For unranked candidates make sure they have extremely high rank.
    data[data == 0] = cnum + 10.
    vote_matrix = np.zeros((cnum, cnum))

    # Get win/loss matrix for each candidate pair
    for i in range(cnum):
        for j in range(i+1, cnum):
            di = data[:, i]
            dj = data[:, j]

            Vij = np.sum(di < dj)
            Vji = np.sum(dj < di)
            vote_matrix[i, j] = Vij
            vote_matrix[j, i] = Vji
    return vote_matrix


def pairwise_scored_matrix(scores):
    """
    Get head-to-head votes for a candidate against another candidate
    given scored voter data

    Parameters
    ----------
    scores : array shaped (a, b)
        Election voter scores, 0 to max.
        Data of candidate ratings for each voter, with

           - `a` Voters represented as each rows
           - `b` Candidates represented as each column.

    Returns
    --------
    out : array shaped (b, b,)
        Vote matrix V[i,j]

        - Total wins for candidate i against candidate j
    """
    data = np.atleast_2d(scores)
    cnum = data.shape[1]
    vote_matrix = np.zeros((cnum, cnum))
    for i in range(cnum):
        for j in range(i+1, cnum):
            di = data[:, i]
            dj = data[:, j]

            Vij = np.sum(di > dj)
            Vji = np.sum(dj > di)
            vote_matrix[i, j] = Vij
            vote_matrix[j, i] = Vji
    return vote_matrix



def smith_set(ranks=None, vm=None, wl=None):
    """
    Compute smith set. Based on Stack-Overflow code.

    From
    https://stackoverflow.com/questions/55518900/seeking-pseudo-code-for-calculating-the-smith-and-schwartz-set
    Retrieved Dec 28, 2019

    Parameters
    -------------
    ranks : array shaped (a, b)
        Election voter rankings, from [1 to b].
        Data composed of candidate rankings, with

           - Voters as the rows
           - Candidates as the columns.

        Use 0 to specify unranked (and therefore not to be counted) ballots.

        - a : number of voters dimension. Voters assign ranks for each candidate.
        - b : number of candidates. A score is assigned for each candidate
              from 0 to b-1.

    vm : array shaped (b, b)
        Votes matrix in favor of ith row candidate against jth column candidate

    wl : array shaped (b, b)
        Win minus Loss matrix

    Returns
    -------
    out : set
        Candidate indices of smith set

    """
    if ranks is not None:
        vote_matrix = pairwise_rank_matrix(ranks)
        win_losses = vote_matrix - vote_matrix.T
    elif wl is not None:
        win_losses = np.array(wl)
    elif vm is not None:
        win_losses = vm - vm.T

    cnum = len(win_losses)
    beats = win_losses > 0
    cannot_beat = ~ beats

    # compute transitive closure of the matrix
    # using Floyd-Warshall algorithm
    for k in range(cnum):
        for i in range(cnum):
            for j in range(cnum):
                cannot_beat[i,j] = cannot_beat[i, j] \
                or (cannot_beat[i, k] and cannot_beat[k, j])

    smith_candidates = []
    for i in range(cnum):
        smith_candidate = {i}
        for j in range(cnum):
            if cannot_beat[i, j]:
                smith_candidate.add(j)
        smith_candidates.append(smith_candidate)

    # Take the smallest candidate
    smith_set1 = min(smith_candidates, key=len)
    return smith_set1


def condorcet_check_one(ranks=None, scores=None):
    """Calculate condorcet winner from ranked data if winner exists.
    Partial election method; function does not handle outcomes when
    condorcet winner is not found.

    Parameters
    ----------
    ranks : array shaped (a, b)
        Election voter rankings, from [1 to b].
        Data composed of candidate rankings, with

           - Voters as the rows
           - Candidates as the columns.

        Use 0 to specify unranked (and therefore not to be counted) ballots.

        - a : number of voters dimension. Voters assign ranks for each candidate.
        - b : number of candidates. A score is assigned for each candidate
              from 0 to b-1.

    Returns
    -------
    out : int
        - Index location of condorcet winner.
        - Return -1 if no condorcet winner found.
    """
    if ranks is not None:
        m = pairwise_rank_matrix(ranks)
    elif scores is not None:
        m = pairwise_scored_matrix(scores)
    else:
        raise ValueError('You must set either argument ranks or scores.'
                         'Both are currently not set as None.')

    win_losses = m - m.T
    beats = win_losses > 0
    cnum = len(m) - 1
    beatnum = np.sum(beats, axis=1)

    i = np.where(beatnum == cnum)[0]
    if len(i) == 0:
        return -1
    elif len(i) == 1:
        return i[0]
    else:
        raise RuntimeError('Something went wrong that should not have happened')


def condorcet_winners_check(ranks=None, matrix=None, pairs=None, numwin=1,
                            full_ranking=False):
    """
    General purpose condorcet winner checker for multiple winners.
    This check does not resolve cycles.

    Parameters
    -----------
    ranks : array shaped (a, b)
        Election voter rankings, from [1 to b].
        Data composed of candidate rankings, with

           - Voters as the rows
           - Candidates as the columns.

        Use 0 to specify unranked (and therefore not to be counted) ballots.

        - a : number of voters dimension. Voters assign ranks for each candidate.
        - b : number of candidates. A score is assigned for each candidate
              from 0 to b-1.

    matrix : array shaped (b, b)
        Win minus Loss margin matrix

    pairs : array shaped (c, 3)
        Win-Loss candidate pairs

        - column 0 = winning candidate
        - column 1 = losing candidate
        - column 2 = margin of victory
        - column 3 = votes for winner

    full_ranking : bool (default False)
        If True evaluate entire ranking of candidates for score output



    Returns
    ------
    winners : array of shape(d,)
        Index locations of each winner.
          - b = `numwin` if no ties detected
          - b > 1 if ties are detected.
          - winners is empty if Condorcet cycle detected

    ties : array of shape (e,)
        Index locations of ties
    scores : array of shape (b,)
        Rankings of all candidates
    """
    if ranks is not None:
        vm = VoteMatrix(ranks=ranks)
        vp = VotePairs(vm.pairs)
        cnum = vm.cnum
    elif matrix is not None:
        vm = VoteMatrix(matrix=matrix)
        vp = VotePairs(vm.pairs)
        cnum = vm.cnum
    elif pairs is not None:
        vm = None
        vp = VotePairs(pairs)
        cnum = int(np.max(vp.pairs[:, 0:2]) + 1)

    else:
        raise ValueError('either ranks, matrix, or pairs must be specified')

    if full_ranking:
        cmax = cnum
    else:
        cmax = numwin

    scores = np.empty(cnum)
    scores[:] = np.nan

    # Everyone ties if cycle detected
    if vp.has_cycle:
        t = np.arange(cmax)
        w = np.array([], dtype=int)

    else:
        for ii in range(cnum):
            iwinners = vp.condorcet_winners
            scores[iwinners] = -ii - 1
            vp = vp.prune_winners()

        scores[np.isnan(scores)] = -ii - 1
        w, t = winner_check(scores, numwin=numwin)

    return w, t, scores


class VoteMatrix(object):
    """
    Pairwise vote information
    
    Parameters
    ----------
    ranks : array shape (a, b)
        Ranking data
    matrix : array shape (b, b)
        Head-to-head vote matrix
        
    """
    def __init__(self, ranks=None, matrix=None):
        if ranks is not None:
            matrix = pairwise_rank_matrix(ranks)
        elif matrix is not None:
            matrix = np.atleast_2d(matrix)
        else:
            s = 'Either matrix or ranks must be defined. Both are None'
            raise ValueError(s)
        self._matrix = matrix

        return

    @property
    def matrix(self):
        return self._matrix


    @lazy_property
    def cnum(self):
        """Number of candidates"""
        return len(self._matrix)


    @lazy_property
    def margin_matrix(self):
        m = self.matrix
        return m - m.T


    @lazy_property
    def pairs(self):
        """array shaped (a, 3)
            Win-Loss candidate pairs

            - column 0 = winning candidate
            - column 1 = losing candidate
            - column 2 = margin of victory
            - column 3 = votes for winner
        """
        # construct win-loss candidate pairs
        win_losses = self.margin_matrix

        cnum = self.cnum

        pairs = []
        for i in range(cnum):
            for j in range(cnum):
                winlosses = win_losses[i, j]
                votes = self.matrix[i, j]
                if winlosses > 0:
                    d = [i, j, winlosses, votes]
                    pairs.append(d)
        return np.array(pairs)


class VotePairsError(Exception):
    """Special votepairs exception for condorcet methods"""
    pass



class VotePairs(object):
    """
    Condorcet calculations for winner-loser pairs in head-to-head matchups.

    """
    def __init__(self, pairs):
        self._pairs = np.atleast_2d(pairs)
        return

    @property
    def pairs(self):
        """array shape (a, 2) : winner loser pairs"""
        return self._pairs


    @property
    def winners(self):
        """array shape (a,) : winners"""
        return self.pairs[:, 0]


    @property
    def losers(self):
        """array shape (a,) : losers"""
        return self.pairs[:, 1]


    @lazy_property
    def condorcet_losers(self):
        """
        Find Condorcet losers in pairs that lose all all other pairs
        """

        pairs = self.pairs
        winners = pairs[:, 0]
        ulosers = np.unique(pairs[:, 1])

        cond_losers =[]
        for iloser in ulosers:
            if iloser not in winners:
                cond_losers.append(iloser)
        return cond_losers


    @lazy_property
    def condorcet_winners(self, ):
        """
        Find Condorcet winners in pairs
        """
        losers = self.losers
        uwinners = np.unique(self.winners).astype(int)

        cond_winners = []
        for iwinner in uwinners:
            if iwinner not in losers:
                cond_winners.append(iwinner)
        return cond_winners





    def copy(self):
        """Return copy of VotePairs"""
        return VotePairs(self.pairs)



    def remove_candidate(self, ilist):
        """Remove candidates from the pairs"""
        iremove = np.zeros(self.pairs.shape[0], dtype=bool)
        for i in ilist:
            i1 = i == self.losers
            i2 = i == self.winners
            iremove = iremove | i1 | i2
        return VotePairs(self.pairs[~iremove])


    def prune_losers(self):
        """
        Prune condorcet losers out of the pairs list

        Returns
        -------
        out : VotePairs
            New pruned pairs

        Raises
        ------
        VotePairsError
            Raised if no Condorcet losers can be pruned.
        """
        cond_losers = self.condorcet_losers
        pair_losers = self.pairs[:, 1]
        if len(cond_losers) == 0:
            raise VotePairsError('No condorcet losers found.')

        if len(self.pairs) == 1:
            return self.copy()

        new = []
        for p_loser, pair in zip(pair_losers, self.pairs):
            if p_loser not in cond_losers:
                new.append(pair)

        newpairs = np.array(new)
        return VotePairs(newpairs)


    def prune_winners(self):
        """
        Prune condorcet winners out of the pairs list
        """
        cwinners = self.condorcet_winners
        return self.remove_candidate(cwinners)


    @property
    def has_cycle(self):
        """
        Determine whether pairs have a Condorcet cycle
        """
        flag = False
        vpairs = self
        cnum = len(vpairs.pairs)
        while cnum > 1:

            try:
                vpairs = vpairs.prune_losers()
            except VotePairsError:
                flag = True
                break
            cnum = len(vpairs.pairs)
        return flag


def has_cycle(pairs):
    """Check if there is a condorcet cycle.

    Parameters
    ---------
    pairs : array shaped (a, 3)
        Win-Loss candidate pairs

        - column 0 = winning candidate
        - column 1 = losing candidate
        - column 2 = margin of victory
        - column 3 = votes for winner

    Returns
    -------
    out : bool
        True if cycle detected. False otherwise
    """

    vp = VotePairs(pairs)
    return vp.has_cycle



class __CycleDetector(object):
    """Base object for detecting condorcet cycles... Might not be working so well

    Parameters
    -----------
    pairs : array shape (a, 3)
        Win-Loss candidate pairs
        - column 0 = winning candidate
        - column 1 = losing candidate
        - column 2 = margin of victory
    """

    def __init__(self, pairs, maxiter=1000):
        self.pairs = pairs
        self.winners = pairs[:, 0]
        self.losers = pairs[:, 1]
        self.pnum = len(pairs)
        self.maxiter = maxiter
        return


    def get_connected_loser_pairs(self, i):
        """
        Retrieve connecting pairs for i-th (winner, loser) pair
        that lose to loser.

        Parameters
        ----------
        i : int
            Index location in pairs of the current pair

        Returns
        --------
        out : array of int
            Index locations of pairs that lose to loser
        """
        base_pair = self.pairs[i]
        loser = base_pair[1]
        locs = np.where(self.winners == loser)[0]
        #logger.debug('locs=%s', locs)
        return list(locs)


    def get_connected_winner_pairs(self, i):
        """
        Retrieve connected pairs than win over the current pair

        Parameters
        ----------
        i : int
            Index location in pairs of the current pair

        Returns
        --------
        out : array of int
            Index locations of pairs that win to winner
        """
        base_pair = self.pairs[i]
        winner = base_pair[0]
        locs = np.where(self.losers == winner)[0]
        #logger.debug('locs=%s', locs)
        return list(locs)



    def any_circuits(self):
        """Determine if there are any closed circuits in graph"""
        for i in range(self.pnum):
            if self.is_circuit(i):
                return True
        return False


    def _is_circuit_recursive(self, start, i=None, branches=None):
        """
        Build beat paths starting with pair i

        Parameters
        ----------
        start : int
            Starting edge
        i : int
            Current edge
        branches : list
            Unexplored branches; indices of edge of each branch

        Returns
        -------
        out : int
            * Start of branch if a closed circuit is found
            * out == -1 if no closed circuit found
        """
        if branches is None:
            branches = []
        if i is None:
            i = start

        ibranches = self.get_connected_loser_pairs(i)
        if start in ibranches:
            return True

        # if no branches we've reached a dead end. Get other branches.
        if len(ibranches) == 0:
            try:
                i = branches.pop(0)
            # if no other branches found, no circuit detected
            except IndexError:
                return False
        # if more branches found, take the first and put the rest in storage
        else:
            i = ibranches.pop(0)
            branches.extend(ibranches)

        return self.is_circuit_recursive(start, i=i, branches=branches)


    def is_circuit(self, start, known_circuits=None):
        """
        Build beat paths starting with pair i

        Parameters
        ----------
        start : int
            Starting edge
        i : int
            Current edge
        known_circuits : list
            Edges that are known to form closed circuits.

        Returns
        -------
        out : int
            * Start of branch if a closed circuit is found
            * out == -1 if no closed circuit found
        """

        if known_circuits is None:
            known_circuits = []

        branches = []
        i = start
        logger.debug('\nCondorcet Cycle Detection\n==========================\n')
        logger.debug('starting pair=%s', i)

        for iterj in range(self.maxiter):

            # Get branches of the current edge
            logger.debug('\n----------------------\nIteration %s\n', iterj)
            logger.debug('Current edge=%s, start=%s', i, start)
            logger.debug('Total branches=\n%s', branches)

            ibranches = self.get_connected_loser_pairs(i)

            logger.debug('connected downstream=\n%s', ibranches)

            if start in known_circuits:
                return known_circuits

            if start in ibranches:
                known_circuits.append(start)
                return known_circuits

            # if no branches we've reached a dead end. Get other branches.
            if len(ibranches) == 0:
                try:
                    i = branches.pop(0)
                    logger.debug('dead end found. Getting branch %d', i)
                # if no other branches found, no circuit detected
                except IndexError:
                    logger.debug('No circuits found')
                    return False

            # if more branches found, take the first and put the rest in storage
            # Set start to the base branch.
            else:
                winbranches = self.get_connected_winner_pairs(i)
                iold = i
                i = ibranches.pop(0)
                logger.debug('connected upstream=\n%s', winbranches)

                branches.extend(ibranches)
                if len(ibranches) > 1:
                    logger.debug('Loser Fork found. Starting at branch %d', i)
                    start = i

                elif len(winbranches) > 1:
                    logger.debug('Win Merge found. Starting at branch %d down %d', iold, i)
                    start = iold
                else:
                    logger.debug('Continuing down branch %d', i)

        raise RuntimeError("Max iteration exceeded, Calculation went wrong")






