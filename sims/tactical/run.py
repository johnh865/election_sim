# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 18:17:37 2020

@author: John
"""
import pdb
import os
import votesim
from  votesim.benchmarks import runtools, tactical, tactical_v2
import definitions

# benchmark = tactical.tactical0()
benchmark = tactical_v2.tactical_v2_1()

METHODS = votesim.votemethods.all_methods
METHODS = [
    
    'smith_minimax',
    'ranked_pairs',
    'irv',
    # # # 'irv_stv',
    'top_two',
    # # # 'rrv',
    # # # 'sequential_monroe',
    'score',
    # # # 'star',
    'maj_judge',
    'smith_score',
    # # 'approval100',
    # # 'approval75',
    'approval50',
    # 'score5',
    # 'score10',
    'star5',
    # 'star10',
    'plurality',
    
    ]
DIRNAME = definitions.DIR_DATA_BENCHMARKS
DIRNAME = os.path.join(DIRNAME, benchmark.name)


def run():    
    os.makedirs(DIRNAME, exist_ok=True)
    df = benchmark.run(METHODS, cpus=4, dirname=DIRNAME)
    return df 
    


    
if __name__ == '__main__':
    try:
        df = run()
    except Exception:
        pdb.post_mortem()
  
    
    
    