# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 10:02:25 2021

@author: John
"""
import pdb
import pandas as pd
import votesim
import os
# from votesim.benchmarks.simpleNd import SimpleThreeWay
from  votesim.benchmarks import runtools, simple
print(os.getcwd())
import definitions

benchmark = simple.simple_base_compare()

METHODS = votesim.votemethods.all_methods
METHODS = [
    
    # 'smith_minimax',
    # 'ranked_pairs',
    # 'irv',
    # # 'irv_stv',
    # 'top_two',
    # 'rrv',
    # 'sequential_monroe',
    # 'score',
    # 'star',
    'maj_judge',
    # 'smith_score',
    # 'approval100',
    # # 'approval75',
    # 'approval50',
    # 'score5',
    # # 'score10',
    # 'star5',
    # # 'star10',
    # 'plurality',
    
    ]
DIRNAME = definitions.DIR_DATA_BENCHMARKS
DIRNAME = os.path.join(DIRNAME, benchmark.name)


def run():    
    os.makedirs(DIRNAME, exist_ok=True)
    benchmark.run(METHODS, cpus=1, dirname=DIRNAME)
    return
    


    
if __name__ == '__main__':
    import logging
    logging.basicConfig()
    logger = logging.getLogger('votesim.benchmarks.simple')
    logger.setLevel(logging.DEBUG)
    run()
    # try:
    #     run()
    # except:
    #     pdb.post_mortem()
        
    
    
    
    
    