# -*- coding: utf-8 -*-

"""
General purpose election running function
"""
import logging
import numpy as np
from votesim.votemethods import  tools
from votesim.votemethods.methodinfo import (
    ranked_methods,
    rated_methods, 
    scored_methods, 
    vote_methods,
    # all_methods,
    )

__all__ = [
        # 'ranked_methods',
        # 'rated_methods',
        # 'scored_methods',
        # 'vote_methods',
        # 'all_methods',
        'eRunner',
        ]


logger = logging.getLogger(__name__)

# ranked_methods = {}
# ranked_methods['smith_minimax'] = condorcet.smith_minimax
# ranked_methods['ranked_pairs'] = condorcet.ranked_pairs
# ranked_methods['irv'] = irv.irv
# ranked_methods['irv_stv'] = irv.irv_stv
# ranked_methods['top_two'] = irv.top2runoff

# rated_methods = {}
# rated_methods['approval100'] = score.approval100
# rated_methods['approval75'] = score.approval75
# rated_methods['approval50'] = score.approval50
# rated_methods['approval25'] = score.approval25
# rated_methods['score5'] = score.score5
# rated_methods['score10'] = score.score10
# rated_methods['star5'] = score.star5
# rated_methods['star10'] = score.star10

# scored_methods = {}
# scored_methods['rrv'] = score.reweighted_range
# scored_methods['seq_monroe'] = score.sequential_monroe
# scored_methods['score'] = score.score
# scored_methods['star'] = score.star
# scored_methods['maj_judge'] = score.majority_judgment
# scored_methods['smith_score'] = condorcet.smith_score

# vote_methods = {}
# vote_methods['plurality'] = plurality.plurality

# all_methods = {}
# all_methods.update(ranked_methods)
# all_methods.update(scored_methods)
# all_methods.update(rated_methods)
# all_methods.update(vote_methods)


class eRunner(object):
    """Run the election & obtain results. For ties, randomly choose winner.
    
    Parameters
    ----------
    etype : str
        Name of election type.
        Mutually exclusive with `method` and `btype`
        Supports the following election types:
          
        - 'approval100' - approval voting, 100% acceptance of nonzero score
        - 'approval50' - approval voting, 50% acceptance of nonzero score
        - 'irv' -- Instant runoff.
        - 'irv_stv' -- Instant runoff with single-transferable vote.
        - 'rrv' -- Reweighted range voting.
        - 'plurality' -- Traditional plurality & Single No Transferable Vote.
        - 'sequential_monroe' -- PR scored method
        - 'score' -- traditional score voting
        - 'smith_minimax' - Smith minimax condorcet voting
        - 'star' -- STAR voting variant of score
        
    method : func
        Voting method function. Takes in argument `data` array shaped (a, b)
        for (voters, candidates) as well as additional kwargs.
        Mutually exclusive with `etype`.
        
        >>> out = method(data, numwin=self.numwinneres, **kwargs)
        
    btype : str
        Voting method's ballot type. 
        Mutually exclusive with `etype`, use with `method`.
        
        - 'rank' -- Use candidate ranking from 1 (first place) to n (last plast), with 0 for unranked.
        - 'score' -- Use candidate integer rating/scored method.
        - 'vote' -- Use traditional, single-selection vote. Vote for one (1), everyone else zero (0).
        - 'rating' -- Use raw ratings data.
    numwinners : int
        Number of winners to consider. Defaults to 1.
    
    ballots :  array shape (a, b)
        Ballots to use in election. 
        
    seed : int or None or array-like
        Seed to input into numpy RandomState.
    rstate : RandomState
        numpy.random.RandomState object 
    

        
    Attributes
    ----------
    winners : array shape (c,)
        Winning candidate indices, including broken ties. 
    winnners_no_ties : array shaped (e,)
        Winning candidate indices without including randomly broken ties
    ties : array shape (d,)
        Tie candidate indices
    output : dict
        Election output
    ballots : array shape (a, b)
        Voter ballots
    btype : str
        Ballot type
    """    
    
    def __init__(self,
                 etype=None, method=None, btype=None,
                 numwinners=1, ballots=None,
                 seed=None, rstate=None, kwargs=None):
        
        logger.debug('eRunner: etype=%s, method=%s', etype, method)
        
        
        if ballots is None:
            raise ValueError("ballots keyword must be specified")
        ballots = np.copy(ballots)
        
        if rstate is None:
            rstate = np.random.RandomState(seed=seed)                
        if kwargs is None:
            kwargs = {}    
            
        ## Run canned election systems with prefilled parameters        
        if method is None:
            if etype in ranked_methods:
                btype = 'rank'
                method = ranked_methods[etype]
                
            elif etype in scored_methods:
                btype = 'score'
                method = scored_methods[etype]
                
            elif etype in rated_methods:
                btype = 'rating'
                method = rated_methods[etype]
                
            elif etype in vote_methods:
                btype = 'vote'
                method = vote_methods[etype]
                                
            else: 
                raise ValueError('%s type not a valid voting method.' % etype)    
                
        # Check if 'seed' is a keyword argument and therefore the voting 
        # method may need random number generation.
        argnum = method.__code__.co_argcount
        fargs = method.__code__.co_varnames[0 : argnum]
        
        if 'rstate' in fargs:
            kwargs['rstate'] = rstate
        elif 'seed' in fargs:
            kwargs['seed'] = seed
        if 'numwin' in fargs:
            kwargs['numwin'] = numwinners
        out1 = method(ballots, **kwargs)       
        # # Run the election method
        # try:     
        #     out1 = method(ballots, numwin=numwinners, **kwargs)            
        # except TypeError:
        #     out1 = method(ballots, **kwargs)      
            
        winners = out1[0]
        ties = out1[1]
        try:
            output = out1[2]        
        except IndexError:
            output = None
        
        ######################################################################
        self.winners_no_ties = winners        
        winners = tools.handle_ties(winners, ties, numwinners, rstate=rstate)        
        
        self.etype = etype
        self.winners = winners
        self.ties = ties
        self.output = output
        self.ballots = ballots
        self.btype = btype
        self._method = method
        self._kwargs = kwargs
        return
    
