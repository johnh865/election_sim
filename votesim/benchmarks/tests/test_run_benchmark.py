# -*- coding: utf-8 -*-
"""Warning! for multiprocess you must use if __name__ == '__main__' """
import os

import votesim
import votesim.benchmarks as bm
# from votesim.utilities import create_dirs

methods = ['plurality', 'star', 'irv']
def main():
    
    newdir = 'output'
    newdir = os.path.join(os.getcwd(), newdir)
    os.makedirs(newdir, exist_ok=True)
    os.chdir(newdir)
    
    print(bm.get_benchmarks())
    # name = 'error-dummy'
    name = 'simple-dummy'
    # name = 'simple-dummy'
    df = bm.run_benchmark(name, methods, cpus=4)[0]
    bm.post_benchmark(name)
    bm.plot_benchmark(name)
    
    keys = list(df.keys())
    
    assert 'output.winner.regret_efficiency_voter' in keys
    assert 'output.candidate.regret_avg' in keys
    
    return keys

if __name__ == '__main__':
    k = main()
    # a = os.getcwd()