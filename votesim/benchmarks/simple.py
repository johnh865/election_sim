"""
Simple benchmarks based on N-dim voter models
"""
import numpy as np

import votesim
import votesim.benchmarks.runtools as runtools
from votesim.models import spatial
import logging

logger = logging.getLogger(__name__)


def simple_model(name, methods, 
                 seed=0,
                 numvoters=100,
                 cnum=3,
                 trialnum=1,
                 ndim=1,
                 stol=1,
                 base='linear'):
    """Simple Election model """

    e = spatial.Election(None, None, seed=seed, name=name)

    v = spatial.Voters(seed=seed, tol=stol, base=base)
    v.add_random(numvoters, ndim=ndim)
    msg_base = 'seed=%s, numvoters=%s, cnum=%s, trial=%s, ndim=%s, stol=%s, base=%s'
    cseed = seed * trialnum
    for trial in range(trialnum):
        logger.debug(
            msg_base, 
            seed, numvoters, cnum, trial, ndim, stol, base)
        c = spatial.Candidates(v, seed=trial + cseed)
        c.add_random(cnum, sdev=1.5)
        e.set_models(voters=v, candidates=c)
        
        # Save parameters
        e.user_data(
                    num_voters=numvoters,
                    num_candidates=cnum,
                    num_dimensions=ndim,
                    voter_tolerance=stol
                    )
        
        for method in methods:
            e.run(etype=method)
            
    return e


# class Simple3Way:
#     name = 'simple-three-way'
#     model = simple_model
#     kwargs = {}    
#     kwargs['name'] = name
#     kwargs['seed'] = np.arange(100)
#     kwargs['numvoters'] = 100
#     kwargs['trialnum'] = 100
#     kwargs['ndim'] = 1
#     kwargs['strategy'] = 'voter'
#     kwargs['stol'] = [.25, 0.5, 1, 1.5, 2, 3]
#     kwargs['cnum'] = 3
    
#     case_args = runtools.CaseGenerator(**kwargs)
#     benchmark =  runtools.CreateBenchmark(name, model, case_args)    



def simple3way():
    
    name = 'simple-three-way'
    model = simple_model
    kwargs = {}    
    kwargs['name'] = name
    kwargs['seed'] = np.arange(100)
    kwargs['numvoters'] = 100
    kwargs['trialnum'] = 100
    kwargs['ndim'] = 1
    kwargs['stol'] = [.25, 0.5, 1, 1.5, 2, 3]
    kwargs['cnum'] = 3
    
    case_args = runtools.CaseGenerator(**kwargs)
    benchmark = runtools.CreateBenchmark(name, model, case_args)    
    return benchmark


def simple_dummy():
    """
    Dummy benchmark for testing purposes. Runs a tiny number of cases.
    """    
    name = 'simple-dummy'
    model = simple_model
    kwargs = {}
    kwargs['name'] = name
    kwargs['seed'] = 0
    kwargs['numvoters'] = 10
    kwargs['trialnum'] = 10
    kwargs['ndim'] = (1, 2)
    kwargs['stol'] = (None, 1.0)
    kwargs['cnum'] = (2, 3, 4)
    case_args = runtools.CaseGenerator(**kwargs)
    benchmark = runtools.CreateBenchmark(name, model, case_args)
    return benchmark


def simple5dim():
    """SimpleNd benchmark #1
    
    Features
    ---------
    
    - Spatial model
    - Normal voter preference distribution
    - 1-5 spatial preference dimensions
    - 2-8 candidates within 1.5 std deviations of voter preferences
    - 2-8 candidates within 3.0 std deviations of voter preferences
    - 2 voter strategies.
    
     #1, 1-5 dimensions, (2-8) candidates x 2'
    """    
    name = 'simple-5dim'
    model = simple_model
    kwargs = {}
    kwargs['name'] = name
    kwargs['seed'] = np.arange(100)
    kwargs['numvoters'] = 101
    kwargs['trialnum'] = 100
    kwargs['ndim'] = np.arange(1, 6)
    kwargs['stol'] = None
    kwargs['cnum'] = [3, 4, 5, 7, 9]
    case_args = runtools.CaseGenerator(**kwargs)
    benchmark = runtools.CreateBenchmark(name, model, case_args)
    return benchmark



def simple_base_compare():
    
    name = 'simple-base-compare'
    model = simple_model
    
    kwargs = {}
    kwargs['name'] = name
    kwargs['seed'] = np.arange(100)
    kwargs['numvoters'] = 51
    kwargs['trialnum'] = 100
    kwargs['ndim'] = [1,2,3]
    kwargs['stol'] = [0.5, 0.75, 1.0]
    kwargs['cnum'] = 5
    kwargs['base'] = ['linear', 'quadratic', 'sqrt']
    case_args = runtools.CaseGenerator(**kwargs)
    benchmark = runtools.CreateBenchmark(name, model, case_args)
    return benchmark



def simple_base_compare_test():
    
    name = 'simple-base-compare-test'
    model = simple_model
    
    kwargs = {}
    kwargs['name'] = name
    kwargs['seed'] = np.arange(10)
    kwargs['numvoters'] = 51
    kwargs['trialnum'] = 10
    kwargs['ndim'] = [1,2,3]
    kwargs['stol'] = [0.5, 1.5, 3.0]
    kwargs['cnum'] = 5
    kwargs['base'] = ['linear', 'quadratic', 'sqrt']
    case_args = runtools.CaseGenerator(**kwargs)
    benchmark = runtools.CreateBenchmark(name, model, case_args)
    return benchmark


    
# class Simple6Dim:
#     """SimpleNd benchmark #1
    
#     Features
#     ---------
    
#     - Spatial model
#     - Normal voter preference distribution
#     - 1-5 spatial preference dimensions
#     - 2-8 candidates within 1.5 std deviations of voter preferences
#     - 2-8 candidates within 3.0 std deviations of voter preferences
#     - 2 voter strategies.
    
#      #1, 1-5 dimensions, (2-8) candidates x 2'
#     """    
#     name = 'simple-6dim'
#     model = simple_model
#     kwargs = {}
#     kwargs['name'] = name
#     kwargs['seed'] = np.arange(100)
#     kwargs['numvoters'] = 100
#     kwargs['trialnum'] = 100
#     kwargs['ndim'] = np.arange(1, 6)
#     kwargs['strategy'] = 'candidate'
#     kwargs['cnum'] = np.arange(3, 9)
#     case_args = runtools.CaseGenerator(**kwargs)
#     benchmark =  runtools.CreateBenchmark(name, model, case_args)


# class SimpleDummy:
#     """
#     Dummy benchmark for testing purposes. Runs a tiny number of cases.
#     """    
#     name = 'simple-dummy'
#     model = simple_model
#     kwargs = {}
#     kwargs['name'] = name
#     kwargs['seed'] = 0
#     kwargs['numvoters'] = 10
#     kwargs['trialnum'] = 10
#     kwargs['ndim'] = (1, 2)
#     kwargs['strategy'] = ('candidate', 'voter')
#     kwargs['cnum'] = (2, 3, 4)
#     case_args = runtools.CaseGenerator(**kwargs)
#     benchmark = runtools.CreateBenchmark(name, model, case_args)


# def _simple3way() -> runtools.CreateBenchmark:
    
#     name = 'simple-three-way'
#     model = simple_model
#     kwargs = {}    
#     kwargs['name'] = name
#     kwargs['seed'] = np.arange(100)
#     kwargs['numvoters'] = 100
#     kwargs['trialnum'] = 100
#     kwargs['ndim'] = 1
#     kwargs['strategy'] = 'voter'
#     kwargs['stol'] = [.25, 0.5, 1, 1.5, 2, 3]
#     kwargs['cnum'] = 3
    
#     case_args = runtools.CaseGenerator(**kwargs)
#     return runtools.CreateBenchmark(name, model, case_args)

# simple3way = _simple3way()


# def _simple6dim() -> runtools.CreateBenchmark:
#     """SimpleNd benchmark #1
    
#     Features
#     ---------
    
#     - Spatial model
#     - Normal voter preference distribution
#     - 1-5 spatial preference dimensions
#     - 2-8 candidates within 1.5 std deviations of voter preferences
#     - 2-8 candidates within 3.0 std deviations of voter preferences
#     - 2 voter strategies.
    
#      #1, 1-5 dimensions, (2-8) candidates x 2'
#     """
#     name = 'simple-6dim'
#     model = simple_model
#     kwargs = {}
#     kwargs['name'] = name
#     kwargs['seed'] = np.arange(100)
#     kwargs['numvoters'] = 100
#     kwargs['trialnum'] = 100
#     kwargs['ndim'] = np.arange(1, 6)
#     kwargs['strategy'] = 'candidate'
#     kwargs['cnum'] = np.arange(3, 9)
#     case_args = runtools.CaseGenerator(**kwargs)
#     return runtools.CreateBenchmark(name, model, case_args)

# simple6dim = _simple6dim()


# def _simple_dummy() -> runtools.CreateBenchmark:
#     """
#     Dummy benchmark for testing purposes. Runs a tiny number of cases.
#     """    
#     name = 'simple-dummy'
#     model = simple_model
#     kwargs = {}
#     kwargs['name'] = name
#     kwargs['seed'] = 0
#     kwargs['numvoters'] = 10
#     kwargs['trialnum'] = 10
#     kwargs['ndim'] = (1, 2)
#     kwargs['strategy'] = ('candidate', 'voter')
#     kwargs['cnum'] = (2, 3, 4)
#     case_args = runtools.CaseGenerator(**kwargs)
#     return runtools.CreateBenchmark(name, model, case_args)
    





