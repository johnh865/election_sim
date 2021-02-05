"""Set up Anaconda path for scripts and to open Jupyter Notebooks"""

import sys
import os

##############################################################################
# Find Anaconda's directory path

match = 'Anaconda'

paths = sys.path
found = False
for path in paths:

    a = path.split(os.sep)
    for ii, ai in enumerate(a):
        if ai.startswith(match):
            found = True
            break
    if found:
        break

s = ('Anaconda not found in sys.path! Anaconda is recommended for beginners in '
     'order to use votesim.')
if not found:
    raise Exception(s)
    
a = a[0 : ii + 1]
anaconda_path = os.sep.join(a)
print('Setting Anaconda path to:')
print(anaconda_path)
print('')

##############################################################################
# Get anaconda environment 

filename = 'environment.bat'

try:
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    env_name = ''
    for line in lines:
        if 'set ENV_NAME=' in line:
            env_name = line.split('=')[1]
            print('Environment name found=', env_name)
            break
    
    if env_name == '':
        print('No environment found in configuration. Just using base')
        env_name = 'base'
except FileNotFoundError:
        print('environment.bat not found. Writing new one')
        env_name = 'base'

newline1 = 'set ENV_NAME=%s' % env_name
newline2 = 'set ANACONDA_PATH=%s' % anaconda_path

with open(filename, 'w') as f:
    f.write(newline1)
    f.write(newline2)
    
    

