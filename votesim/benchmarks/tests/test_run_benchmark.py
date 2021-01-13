# -*- coding: utf-8 -*-
"""Warning! for multiprocess you must use if __name__ == '__main__' """
import os

import votesim
import votesim.benchmarks as bm

# from votesim.utilities import create_dirs

methods = ['irv', 'plurality', 'star', 'irv']
def test_main():
    
    newdir = 'output'
    newdir = os.path.join(os.getcwd(), newdir)
    os.makedirs(newdir, exist_ok=True)
    os.chdir(newdir)
    
    # print(bm.get_benchmarks())
    # name = 'error-dummy'
    # name = 'simple-dummy'
    
    benchmark = bm.simple.simple_dummy()
    df = benchmark.run(methods, cpus=1,)
    df = benchmark.run(methods, cpus=2,)
    # name = 'simple-dummy'
    # df = bm.run_benchmark(name, methods, cpus=4)[0]
    # bm.post_benchmark(name)
    # bm.plot_benchmark(name)
    
    keys = list(df.keys())
    
    assert 'output.winner.regret_efficiency_voter' in keys
    assert 'output.candidates.regret_avg' in keys
    
    return keys

# if __name__ == '__main__':
#     k = test_main()
    # a = os.getcwd()
    
    
if __name__ == '__main__':
    test_main()
