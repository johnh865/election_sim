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
    name = 'error-dummy'
    # name = 'simple-dummy'
    # name = 'simple-dummy'
    bm.run_benchmark(name, methods, cpus=8)
    bm.post_benchmark(name)
    bm.plot_benchmark(name)
    


if __name__ == '__main__':
    main()
    # a = os.getcwd()