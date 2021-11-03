# -*- coding: utf-8 -*-
"""Define project path definitions
"""

import os
from os.path import dirname, join

DIR_SIMS = dirname(dirname(__file__))
DIR_DATA_BENCHMARKS =  join(DIR_SIMS, 'output', 'benchmarks')

