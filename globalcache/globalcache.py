# -*- coding: utf-8 -*-
"""
Globalcache constructs a memoization/caching decorator you can use to store
results in IPython or Spyder globals(). This lets you re-run a script and 
skip the heavy computing if the result has already been processed, when you
make small changes to your script. 


Requirements
-----------
Globalcache only works with the option "Run in console's namespace instead of 
an empty one". Or when Spyder calls `spydercustomize.runfile`, set 

>>> spydercustomize.runfile(current_namespace = True)

Features
---------

- Cache expensive output from main script; re-run the script quickly when you need to 
  debug and change later parts of the script while preserving beginning parts

- Cache is dependent on function inputs. Globalcache knows when you change 
  the arguments and therefore need to re-run expensive code. 

- Use argument aliases to identify and cache functions with unhashable arguments. 



Example Usage
--------------

Decorate a function you want to cache

    >>> import globalcache
    >>> import numpy as np
    >>>
    >>> @globalcache.cache_decorate('test1')
    >>> def test1(n):
    >>>     array = np.arange(n)
    >>>     return array
    

Decorate another function; but this function has no hashable arguments.
Create new keyword where you can input a hashable identifier.  

    >>> @globalcache.cache_decorate('test2', alias_key='cache_alias')
    >>> def test2(array):
    >>>     return array * 4


Initialize the global cache using globals()
    
    >>> c = globalcache.create(globals(), 'my-cache', maxsize=5)

Run the cached function test1.
    
    >>> num = 50
    >>> a = test1(num)


To cache test2, use the alias_key keyword to identify cached results. test2 
is dependent on results from test1, so the alias_key can use test1's arguments. 
    
    >>> b = test2(a, cache_alias=num)


You can also create new cache'd functions from old functions

    >>> def func1(x):
    >>>    return x + 2
    >>>
    >>> func2 = globalcache.cache_decorate('func1')(func1)
    


Globals
----------
CACHE_INFO : dict
    CACHE_INFO stores cache'd data. It has the following keys:
        
        GLOBAL_CACHE : dict
            Stores cache of each script. Each key of GLOBAL_CACHE corresponds
            to the path of the source script file that has been cache'd. 
            The values of each key correspond to the session's globals(). 
        ENABLE : bool (default False)
            Enable (True) or disable (False) the global cache.
        SIZE : int
            Max number of previous output to keep for each cached function.



Created on Thu Jun 13 10:41:26 2019

@author: jhuang
"""
from collections import OrderedDict
from functools import wraps
import inspect

CACHE_INFO = {}
CACHE_INFO['GLOBAL_CACHE'] = {}
CACHE_INFO['SIZE'] = {}
CACHE_INFO['ENABLE'] = False


def create(g, name='GLOBAL_CACHE', maxsize=1, reset=False, enable=True):
    """
    Create a global dictionary cache.
    
    Parameters
    -----------
    g : dict from globals()
        Input globals() from main script which you want to control cache from.
    name : str
        Name of the global cache you are creating
    maxsize : int
        Max number of previous output to keep for each cached function. 
        Defaults to 1. 
    reset : bool (default False)
        If True, resets the cache 
        
        
    Returns
    ------
    d : dict
        global cache created inside globals()
        
    """
    sizename = name + "_SIZE"
    name = "__" + name + "__"
    
    if (name not in g) or reset:
        g[name] = {}    
        g[sizename] = {}
        
        
        
    
    CACHE_INFO['GLOBAL_CACHE'] = g[name]
    CACHE_INFO['SIZE'] =  g[sizename]
    CACHE_INFO['ENABLE'] = enable
    return g[name]




def reset():
    """Reset the global cache"""
    del CACHE_INFO['GLOBAL_CACHE']
    del CACHE_INFO['SIZE']
    return


def disable():
    CACHE_INFO['ENABLE'] = False
    
    
def enable():
    CACHE_INFO['ENABLE'] = True
    
    

def cache_decorate(name, alias_key='cache_alias', size=1):
    """Decorator used to cache/memoize a function. You must assign
    a unique name to the function to store into the cache.
    
    This decorator also adds a keyword argument to the target function that can be 
    used to input hashable arguments that can be used for lookup. 
    
    Parameters
    -----------
    name : str
        Unique name used to store function output.
        
    alias_key : str, default "cache_alias"
        New keyword name.  cache_decorate gives the function it decorates a
        new keyword argument. alias_key specifies the name of the new keyword. 
        The new keyword is used to input in an alternative argument 
        alias. 
        
    Returns
    --------
    wrapper : 
        A new function decorator that supports caching.
        
    """

    def wrapper(fn):            

        
        module = inspect.getsourcefile(fn)

        @wraps(fn)
        def newfunc(*args, **kwargs):
            ########################################
            ### Run the function normally if cache not enabled. Get rid of the alias_key.
            if not CACHE_INFO['ENABLE']:
                try:
                    kwargs.pop(alias_key)
                except KeyError:
                    pass
                
                return fn(*args, **kwargs)            
            
            ### Construct keys for dictionary read/access
            try:
                key = kwargs.pop(alias_key)
            except KeyError:
                key = (args, frozenset(kwargs.items()))
            
            ########################################
            ### Retrieve global cache. 
            g = CACHE_INFO['GLOBAL_CACHE']
            gsize = CACHE_INFO['SIZE']
                    
            if module not in g:
                g[module] = {}
                gsize[module] = {}        
                    
            module_dict = g[module]
            maxsize_dict = gsize[module]    
            maxsize_dict[name] = size            
            
            ########################################
            ### Retrieve function cache
            try:
                func_dict = module_dict[name]
            except KeyError:
                func_dict = OrderedDict()
                module_dict[name] = func_dict
                
            ### Get cache size limit
            maxsize = maxsize_dict[name] 
                
            ### Get value of function dictionary
            try:
                return func_dict[key]
            except KeyError:
                value = fn(*args, **kwargs)
                func_dict[key] = value
                
                if len(func_dict) > maxsize:
                    func_dict.popitem(False)
                return value            
                    
        return newfunc    
    
    return wrapper


#def _run_func(name, func, key, args, kwargs):
#    """
#    Run function with cache on.
#    
#    Parameters
#    ----------
#    name : str
#        Name of function
#    func : function
#        Function to decorate
#    key : Hashable
#        Key signature of arguments
#    args : tuple
#        Positional arguments for function
#    kwargs : dict
#        Keyword arguments for function
#    
#    """
#    
#    # Get name of function as source file path + name
#    
#    if not CACHE_INFO['ENABLE']:
#        return func(*args, **kwargs)
#    
#    module = inspect.getsourcefile(func)
#    
#    gdict = CACHE_INFO['GLOBAL_CACHE'][module]
#    maxsize = CACHE_INFO['SIZE'][module]    
#    
##    self.lastargs[name] = key
#    
#    # Get dictioary where function is stored
#    try: 
#        func_dict = gdict[name]
#
#    except KeyError:
#        func_dict = OrderedDict()
#        gdict[name] = func_dict
#        
#    # Get value of function dictionary
#    try:
#        return func_dict[key]
#    except KeyError:
#        value = func(*args, **kwargs)
#        func_dict[key] = value
#        
#        if len(func_dict) > maxsize:
#            func_dict.popitem(False)
#        return value
#    
#    
#









