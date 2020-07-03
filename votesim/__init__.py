"""
Voting simulation and benchmarking module
"""

# from . import logconfig

# from . import votesystems
# from . import models
# from . import utilities
# from . import metrics
# from . import benchmarks

__all__ = [
    'votesystems',
    'models',
    'utilities',
    'metrics',
    'benchmarks',
    'definitions',
    'plots',
    'post',
    ]


from votesim import (
    votesystems,
    models,
    utilities,
    metrics,
    benchmarks,
    definitions,
    plots,
    post,
    )


#from votesim import logconfig as _lc
#logSettings = _lc.LogSettings()


