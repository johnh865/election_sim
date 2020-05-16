"""
Voting simulation and benchmarking module
"""

# from . import logconfig

# from . import votesystems
# from . import models
# from . import utilities
# from . import metrics
# from . import benchmarks

from votesim import (
    votesystems,
    models,
    utilities,
    metrics,
    benchmarks,
    definitions,
    plots
    )


from votesim import logconfig as _lc
logSettings = _lc.LogSettings()
