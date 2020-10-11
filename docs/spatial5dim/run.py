import pandas as pd
import votesim
import os
# from votesim.benchmarks.simpleNd import SimpleThreeWay
from  votesim.benchmarks import runtools, simple

benchmark = simple.simple5dim()

METHODS = votesim.votemethods.all_methods
METHODS = [
    
    # 'smith_minimax',
    # 'ranked_pairs',
    # 'irv',
    # # 'irv_stv',
    # 'top_two',
    # # 'rrv',
    # # 'sequential_monroe',
    # # 'score',
    # # 'star',
    # 'maj_judge',
    # 'smith_score',
    # 'approval100',
    # 'approval75',
    # 'approval50',
    'approval25',
    # 'score5',
    # # 'score10',
    # 'star5',
    # # 'star10',
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
  
    
    
    
    
    