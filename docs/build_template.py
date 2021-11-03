# -*- coding: utf-8 -*-
"""
Build template .rst file for aqolt modules.

Created on Fri Aug 17 10:53:21 2018

@author: jhuang
"""

import builder
import importlib
import os

def module_filter(modules):
    """Filter out modules that start with test or _"""
    modules = []
    for module in modules:
        name = module.__name__
        name_arr = name.split('.')
        for n in name_arr:
            if n.startswith('test'):
                remove = True
                break
            elif n.startswith('_'):
                remove = True
                break
            else:
                remove = False
                
        if not remove:
            modules.append(module)
    return modules
    
###################################################
## Built .rst files from rst template

PACKAGE_NAME = 'votesim'
TEMPLATE_PATH = './_templates/MODULE_TEMPLATE.txt'
DIRNAME = 'package'

package = importlib.import_module(PACKAGE_NAME)
with open(TEMPLATE_PATH) as f:
    tstring = f.read()
    
modules = builder.get_modules(package, excludes=('_*', 'test*'))

for module in modules:
    
    newpath = module.__name__ + '.rst'
    newpath = os.path.join(DIRNAME, newpath)
    print( newpath)
    lines = builder.build_module_rst(tstring, module)
    with open(newpath, 'w') as f:
        f.writelines(lines)
        
        
###################################################
## Copy installation instructions
# os.chdir('..')
# os.chdir('..')

# source = 'aqolt/READERME.rst.txt'
# dest = 'docs/source/INSTALL.rst'

# shutil.copy(source, dest)


    
