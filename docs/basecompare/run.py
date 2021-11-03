import logging
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
    
    'smith_minimax',
    'ranked_pairs',
    'irv',
    # 'irv_stv',
    'top_two',
    # 'rrv',
    # 'sequential_monroe',
    # 'score',
    # 'star',
    'maj_judge',
    'smith_score',
    'approval100',
    # 'approval75',
    'approval50',
    'score5',
    # 'score10',
    'star5',
    # 'star10',
    'plurality',
    
    ]

# METHODS = ['score5']
DIRNAME = definitions.DIR_DATA_BENCHMARKS
DIRNAME = os.path.join(DIRNAME, benchmark.name)


def run():    
    os.makedirs(DIRNAME, exist_ok=True)
    benchmark.run(METHODS, cpus=4, dirname=DIRNAME)
    return
    


    
if __name__ == '__main__':
    # logging.basicConfig()
    # logger = logging.getLogger('votesim.votemethods.voterunner')
    # logger.setLevel(logging.INFO)
    run()
  
    
    
    
    
    