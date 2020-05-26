# -*- coding: utf-8 -*-
"""
Build template .rst file 

Created on Fri Aug 17 10:53:21 2018

@author: jhuang
"""
from __future__ import (print_function, division, absolute_import)

import fnmatch
import inspect
import pkgutil
import os
import sys
from os.path import split, join, basename, dirname, splitext
import shutil
import jinja2

import votesim
from votesim.utilities import detectfiles

def get_module_classes(module):
    """Get all classes found in module"""
    modulename = module.__name__
    
    clsmembers = inspect.getmembers(module, inspect.isclass)
    clsmodules = [t[1].__module__ for t in clsmembers]
    new = []
    for member, module in zip(clsmembers, clsmodules):
        if module == modulename:
            new.append(member[-1])
            
    return new


def get_module_funcs(module):
    """Get all functions found in module"""
    modulename = module.__name__
    
    
    clsmembers = inspect.getmembers(module, inspect.isfunction)
    clsmodules = [t[1].__module__ for t in clsmembers]
    new = []
    for member, module in zip(clsmembers, clsmodules):
        if module == modulename:
            new.append(member[-1])
    return new



def get_class_attr_names(clss):
    d = clss.__dict__
    cname = clss.__name__
    new = []
    for key in d:
        if key.startswith('_'):
            pass
        else:
            name = cname + '.' + key
            new.append(name)
            
    return new


def get_names(a):
    """Get names from list of objects"""
    names = []
    for ai in a:
        module = ai.__module__
        name = ai.__name__
        name1 = module + '.' + name
        name1 = name
        names.append(name1)
    return names

#
#def get_module_class_names(modules):
#    classes = get_module_classes(modules)
#    names = []
#    for c in classes:
#        module = c.__module__
#        name = c.__name__
#        name1 = module + '.' + name
#        name1 = name
#        names.append(name1)
#    return names
#
#
#def get_module_func_names(modules):
#    classes = get_module_funcs(modules)
#    names = []
#    for c in classes:
#        module = c.__module__
#        name = c.__name__
#        name1 = module + '.' + name
#        name1 = name
#        names.append(name1)
#    return names


def _get_modules(package, excludes=()):
    """Return modules imported from top package"""

    pname = package.__name__
        
        
    names = [pname]
    modules = [package]
    
    try:
        path = package.__path__
    except AttributeError:
        return names, modules
    
    for loader, name, ispkg in pkgutil.walk_packages(path):
        
        fullname = pname + '.' + name
        skip = False
        for pattern in excludes:
            if fnmatch.fnmatchcase(name, pattern):
                skip = True
                print('Package %s is excluded by user' % fullname)   
                break
        if skip:
            continue
        
        try:
            __import__(fullname)
            failed = False
        except Exception as e:
            failed = True
            print('Failed to import %s -- %s' % (fullname, e))
        
        if not failed:
            module1 = getattr(package, name)
            names.append(fullname)
            modules.append(module1)
            
            if ispkg:
                names1, modules1 = _get_modules(module1, excludes=excludes)            
                names.extend(names1)
                modules.extend(modules1)            
                
    return names, modules


def get_modules(package, excludes=()):
    """Retrieve modules from a package
    
    Paramters
    -----------
    package : module
        Module to search for submodules
    excludes : list
        Name patterns to exclude
    
    """
    return _get_modules(package, excludes)[1]

def get_module_names(package, excludes=()):
    return _get_modules(package, excludes)[0]


def filter_private_names(names):
    new = [n for n in names if (not n.startswith('_'))]
    return new
            



def get_scripts(basepath):
    sources = detectfiles(scriptdir, '*.py')
    
    modules = []
    names = []
    for s in sources:
        dir_bname = basename(dirname(s))
        dirpath = os.path.abspath(dirname(s))
        
        if not dir_bname.startswith('_'):
            name = splitext(basename(s))[0]
            newname = s.replace('\\', '.')
            newname = newname.replace('/', '.')
            newname = splitext(newname)[0]
            
            sys.path.insert(0, dirpath)
            try: 
                module = __import__(name)
                modules.append(module)
                names.append(newname)
            
            except Exception as e:
                print('Failed to import %s -- %s' % (name, e))
            sys.path.pop(0)
            
    return names, modules
                
                


def build_rst(tpath, module):
    """
    Create .rst file using customized generator of documentation that
    detects particular keywords in a template file. The keywords trigger Python
    to find all classes/functions/modules in the specified module. 
    
    Parameters
    -----------
    tpath : str
        Path to template rst file.
    module : Module
        Module to import and construct documentation for.
        
    
    Returns
    --------
    lines : list
        List of str that will make up a new file to create.
        
    """
    with open(tpath, 'r') as f:
        lines = f.readlines()
    classes = get_module_class_names(module)
    functions = get_module_func_names(module)
    
    classes = filter_private_names(classes)
    functions = filter_private_names(functions)
    
    modulename = module.__name__
    
    key_name_module = '"TEMPLATE MODULE NAME"'
    

    
    key_start_class = '"TEMPLATE START CLASS LOOP"'
    key_name_class = '"TEMPLATE CLASS NAME"'
    key_end_class = '"TEMPLATE END CLASS LOOP"'
    
    key_start_func = '"TEMPLATE START FUNC LOOP"'
    key_name_func = '"TEMPLATE FUNC NAME"'
    key_end_func = '"TEMPLATE END FUNC LOOP"'
    
    
    # Function to repeat class and function definitions from template.
    def process_repeat_keys(lines, key):
        """Construct text lines to generate repeated class or function
        defintions 
        """
        if key == key_name_class:
            names = classes
        elif key == key_name_func:
            names = functions
        
        newlines1 = []
        for name in names:
            for line in lines:
                line1 = line.replace(key, name)
                newlines1.append(line1)
        return newlines1
    
    # Process the class and function repetitions.
    in_repeat = False
    repeated_lines = []
    newlines = []
    
    
    # First only detect module name 
    lines1 = []
    for line in lines:
        if key_name_module in line:
            line = line.replace(key_name_module,
                                modulename)    
            lines1.append(line)
        else:
            lines1.append(line)
    
    
    # Now loop to find repeated function/class names
    for line in lines1:
        line0 = line.strip()
        # Check start/end of class loop
        if line0.startswith(key_start_class):
            in_repeat = True
            
        elif line0.startswith(key_end_class):
            in_repeat = False
            repeated_lines = process_repeat_keys(repeated_lines,
                                                 key_name_class)
            newlines.extend(repeated_lines)
            repeated_lines = []
        
        # Check start/end of function loop
        elif line0.startswith(key_start_func):
            in_repeat = True
            
        elif line0.startswith(key_end_func):
            in_repeat = False            
            repeated_lines = process_repeat_keys(repeated_lines,
                                                 key_name_func)
            newlines.extend(repeated_lines)
            repeated_lines = []

        
        else:
            # Add lines to repeated block
            if in_repeat:
                repeated_lines.append(line)
                
            # Check module name    
#            elif key_name_module in line0:
#            
#                line = line.replace(key_name_module,
#                                    modulename)
#                newlines.append(line)
            else:
                newlines.append(line)
        

    return newlines



def build_module_rst(fstr, module):
    """
    Create .rst file using customized generator of documentation using
    jinja2.
    
    Parameters
    -----------
    fstr : str
        Template string
    module : Module
        Module to import and construct documentation for.
        
    
    Returns
    --------
    lines : list
        List of str that will make up a new file to create.    
    """
    
    modulename = module.__name__


    classes = get_module_classes(module)
    functions = get_module_funcs(module)
    
    class_names = get_names(classes)
    func_names = get_names(functions)
    
    classes = dict(zip(class_names, classes))
    functions = dict(zip(func_names, functions))
    
    for k in list(classes.keys()):
        if  k.startswith('_'):
            classes.pop(k)
            
    for k in list(functions.keys()):
        if  k.startswith('_'):
            functions.pop(k)    
    

    class_attrs = {}
    for name, clss in classes.items():
        class_attrs[name] = get_class_attr_names(clss)
        
    t = jinja2.Template(fstr)
    env = jinja2.Environment(trim_blocks=True)
    newlines  = t.render(
                    classes=classes,
                    functions=functions,
                    members=class_attrs,
                    module_name=modulename
                 )
    return newlines
        
    



