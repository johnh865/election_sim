# -*- coding: utf-8 -*-

import os
from os.path import join, dirname
import votesim

DIRNAME = dirname(__file__)
DIRNAME_SIMS = join(DIRNAME, 'sims')

if __name__ == '__main__':
    dirname1 = join(DIRNAME_SIMS, 'simple3way')
    os.chdir(dirname1)
    exec(open('run.py').read())
    exec(open('plot.py').read())
    
    dirname1 = join(DIRNAME_SIMS, 'spatial5dim')
    os.chdir(dirname1)
    exec(open('run.py').read())
    exec(open('plot.py').read())
        
    dirname1 = join(DIRNAME_SIMS, 'tactical')
    os.chdir(dirname1)
    exec(open('run.py').read())
    exec(open('plot.py').read())
    
    
    
    
