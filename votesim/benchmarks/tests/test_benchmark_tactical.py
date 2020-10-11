# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 01:19:07 2020

@author: John
"""
import pdb
import votesim
import logging
import numpy as np
from votesim.benchmarks import tactical


def test_model():
    methods = ['plurality', 'irv', 'score']
    name = 'tactical-model-1'
    e = tactical.tactical_model(name, methods, seed=10)
    df = e.dataframe()
    return df


def test_benchmark():
    methods = ['plurality', 'irv', 'score']
    benchmark = tactical.tactical_dummy()
    df = benchmark.run(methods, cpus=1,)
    
    # test re-run capability
    e2 = benchmark.rerun(index=0, df=df)
    
    # Check to make sure outputs of re-run are the same. 
    s1 = df.loc[0]
    s2 = e2.result.dataseries()
    for key in s2.keys():
        print(key, '=', s1[key])
        assert np.all(s1[key] == s2[key])
    
    return df


if __name__ == '__main__':
    df1 = test_model()
    logging.basicConfig()
    logger = logging.getLogger('votesim.utilities.recorder')
    logger.setLevel(logging.DEBUG)
    
    # try:
    df2 = test_benchmark()
    # except Exception:
    #     pdb.post_mortem()
    
