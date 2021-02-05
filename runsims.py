# -*- coding: utf-8 -*-

import os
import sys
from os.path import join, dirname
import votesim
import subprocess





DIRNAME = dirname(__file__)
DIRNAME_SIMS = join(DIRNAME, 'sims')
dirname1 = join(DIRNAME_SIMS, 'simple3way')

sys.path.insert(0, dirname1)
import definitions



if __name__ == '__main__':
    dirname1 = join(DIRNAME_SIMS, 'simple3way')
    os.chdir(dirname1)
    # votesim.utilities.misc.execfile('run.py')
    # votesim.utilities.misc.execfile('plot.py')
    
    print('Running run.py for simple3way')
    exec(open('run.py').read(), globals(), locals())
    print('Running plot.py for simple3way')
    exec(open('plot.py').read(), globals(), locals())
    
    
    dirname1 = join(DIRNAME_SIMS, 'spatial5dim')
    os.chdir(dirname1)
    print('Running run.py for spatial5dim')
    exec(open('run.py').read(), globals(), locals())
    print('Running plot.py for spatial5dim')
    exec(open('plot.py').read(), globals(), locals())
        
    dirname1 = join(DIRNAME_SIMS, 'tactical')
    os.chdir(dirname1)
    print('Running run.py for tactical')
    exec(open('run.py').read(), globals(), locals())
    print('Running plot.py for tactical')
    exec(open('plot.py').read(), globals(), locals())
    
    
    
