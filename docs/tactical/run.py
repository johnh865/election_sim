# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 18:17:37 2020

@author: John
"""
import os
import votesim
from  votesim.benchmarks import runtools, tactical

benchmark = tactical.tactical0()

METHODS = votesim.votemethods.all_methods
METHODS = [
    
    # 'smith_minimax',
    # 'ranked_pairs',
    'irv',
    # 'irv_stv',
    # 'top_two',
    # 'rrv',
    # 'sequential_monroe',
    # 'score',
    # 'star',
    # 'maj_judge',
    # 'smith_score',
    # 'approval100',
    # 'approval75',
    # 'approval50',
    # 'score5',
    # 'score10',
    # 'star5',
    # 'star10',
    # 'plurality',
    
    ]
DIRNAME = votesim.definitions.DIR_DATA_BENCHMARKS
DIRNAME = os.path.join(DIRNAME, benchmark.name)


def run():    
    os.makedirs(DIRNAME, exist_ok=True)
    benchmark.run(METHODS, cpus=8, dirname=DIRNAME)
    return
    


    
if __name__ == '__main__':
    run()
  
    
    
    