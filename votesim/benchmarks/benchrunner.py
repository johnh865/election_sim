# -*- coding: utf-8 -*-

import os
import votesim

import pandas as pd
import numpy as np

from . import simpleNd
from . import error_v1

_benchmarks = [
    simpleNd.SimpleDummy,
    simpleNd.Simple1,
    simpleNd.SimplePop,
    error_v1.ErrorDummy,
    error_v1.Error1
    ]

benchmarks = {}
for b in _benchmarks:
    benchmarks[b.name] = b
    

#benchmarks['voter_error'] = voter_error.run

def get_benchmarks():
    """Retrieve available benchmarks"""
    return list(benchmarks.keys())


def run_benchmark(name, methods, cpus=1, kwargs=None):
    """
    Run benchmark
    
    Parameters
    ----------
    name : str
        Name of benchmark
    methods : list of str
        List of voting systems to assess
    cpus : int
        Number of processes to spawn
    kwargs : dict
        Addictional keyword arguments for benchmark
    """
    if kwargs is None:
        kwargs = {}
        
    func = benchmarks[name].run
    return func(methods, cpus=cpus, **kwargs)


def post_benchmark(name, kwargs=None):
    """
    Postprocess benchmark
    """
    if kwargs is None:
        kwargs = {}
        
    post = benchmarks[name].post
    return post(**kwargs)


def plot_benchmark(name, kwargs=None):
    if kwargs is None:
        kwargs = {}
    p = benchmarks[name].plot
    return p(**kwargs)












