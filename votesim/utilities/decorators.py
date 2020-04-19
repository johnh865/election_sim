"""
Collection of utilities such as memoization, automatic property storage, etc 
"""
from __future__ import print_function, absolute_import, division

from functools import wraps, partial
import logging
from votesim.utilities import misc



logger = logging.getLogger(__name__)
 
    


class memoize:
    """
    Decorator used to store past calls.
    """    
    def __init__(self, function):
        self.function = function
        self.memoized = {}
    
    
    def __call__(self, *args, **kwargs):
      
        key = (args, frozenset(kwargs.items())) 
        try:
            return self.memoized[key]
        except KeyError:
            self.memoized[key] = self.function(*args, **kwargs)
            return self.memoized[key]



class method_memoize(object):
    """cache the return value of a method
    
    This class is meant to be used as a decorator of methods. The return value
    from a given method invocation will be cached on the instance whose method
    was invoked. All arguments passed to a method decorated with memoize must
    be hashable.
    
    If a memoized method is invoked directly on its class the result will not
    be cached. Instead the method will be invoked like a static method:
    class Obj(object):
        @memoize
        def add_to(self, arg):
            return self + arg
    Obj.add_to(1) # not enough arguments
    Obj.add_to(1, 2) # returns 3, result is not cached
    """
    def __init__(self, func):
        self.func = func
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.func
        return partial(self, obj)
    def __call__(self, *args, **kw):
        obj = args[0]
        try:
            cache = obj.__cache
        except AttributeError:
            cache = obj.__cache = {}
        key = (self.func, args[1:], frozenset(kw.items()))
        try:
            res = cache[key]
        except KeyError:
            res = cache[key] = self.func(*args, **kw)
        return res





#
#def lazyprop(fn):
#    """
#    Decorator used to cache property results
#    
#    From stack overflow. Author Mike Boers
#    https://stackoverflow.com/questions/3012421/python-memoising-deferred-lookup-property-decorator
#    """
#    
#    attr_name = '_lazy_' + fn.__name__
#    @property
#    def _lazyprop(self):
#        if not hasattr(self, attr_name):
#            setattr(self, attr_name, fn(self))
#        return getattr(self, attr_name)
#    return _lazyprop
#    
    
### Lazy Property decorator
# Property name to hold all lazy data
_data_holder_attr = '_cache_properties'


def clean_lazy_properties(instance):
    '''Clean all lazy properties'''
    setattr(instance, _data_holder_attr, {})
    
    
def clean_some_lazy_properties(instance, names):
    """Clean properties in iterable names"""
    try:
        cache = getattr(instance, _data_holder_attr)
    except AttributeError:
        return
    
    if isinstance(names, str):
        names = [names]
        
    for name in names:
        try:
            del cache[name]
        except KeyError:
            pass
        
    setattr(instance, _data_holder_attr, cache)
    return
    
    
def modify_lazy_property(instance, name, value):
    """Modify a lazy property"""
    cache = getattr(instance, _data_holder_attr)
    cache[name] = value
    setattr(instance, _data_holder_attr, cache)
    return
    


def lazy_property(fn):
    """
    Version of lazy_property by John Huang.
    
    Decorator used to cache property results into dictionary.
    The cache can be clered using clean_lazy_properties.
    """
    
    cache_name = _data_holder_attr
    attr_name = fn.__name__
    
    def get_cache(instance):
        if not hasattr(instance, cache_name):
            setattr(instance, cache_name, {})
        return getattr(instance, cache_name)
    
    @property
    @wraps(fn)
    def get_attr(self):
        cache = get_cache(self)
        if attr_name not in cache:
            cache[attr_name] = fn(self)
        return cache[attr_name]
    
    return get_attr



def lazy_property2(name=_data_holder_attr):
    """
    Version of lazy_property by John Huang.
    
    Decorator used to cache property results into dictionary.
    The cache can be clered using clean_lazy_properties.
    
    Decorator must be called as a function. 

    Parameters
    ----------
    name : str
        Name of cache dictionary
        
    
    
    Example
    ---------
    >>> class class1(object):
    >>>     @lazy_property2('my_cache')
    >>>     def property(self):
    >>>         x = 2.0
    >>>         return x 
    
    """
    
    def decorator(fn):
        cache_name = name
        attr_name = fn.__name__
        
        def get_cache(instance):
            if not hasattr(instance, cache_name):
                setattr(instance, cache_name, {})
            return getattr(instance, cache_name)
    
        @property
        @wraps(fn)
        def get_attr(self):
            cache = get_cache(self)
            if attr_name not in cache:
                cache[attr_name] = fn(self)
            return cache[attr_name]
        return get_attr
    return decorator
            
        


                 

