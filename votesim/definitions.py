# -*- coding: utf-8 -*-
"""Define project path definitions
"""

import os



DIR_MODULE = os.path.abspath(os.path.dirname(__file__))
DIR_PROJECT = os.path.dirname(DIR_MODULE)

DIR_REPORTS = os.path.join(DIR_PROJECT, 'reports')
DIR_DATA_BENCHMARKS =  os.path.join(DIR_PROJECT, 'data', 'benchmarks')
DIR_DATA = os.path.join(DIR_PROJECT, 'data')