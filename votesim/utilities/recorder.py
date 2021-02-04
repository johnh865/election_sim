"""
Decorator and storage class used to record object method calls and re-call them
when desired.

"""
# -*- coding: utf-8 -*-
import logging
import math
from functools import wraps
from votesim.utilities import misc

logger = logging.getLogger(__name__)

def record_actions(name='_method_records', replace=False, exclude=()):
    """
    Decord used to record method actions in object created in parent object.
    
    Parameters
    -----------
    name : str (default '_method_records')
        Name of RecordActionCache used to record method actions. 
        This shall be created in the parent object.
        
    replace : bool (default = False)
        - If False, append to the previously recorded arguments for the method
        - If True, replace the previous recorded arguments for the method.
        
    exclude : list[str]
        Arguments to exclude from record.
        
    Returns
    -------
    out : decorator
        Function used to decorate class methods 
        
    Reset Record
    -------------
    Reset the record for object `a` using:
        
    >>> del a._method_records        
    """
    def decorator(fn):
        """Cache a method's arguments in a dict created in parent object"""
        
        funcname = fn.__name__    
        varnames = fn.__code__.co_varnames
        
        def get_cache(instance):
            """Retrieve records cache of object"""
            if not hasattr(instance, name):
                cache = RecordActionCache()
                setattr(instance, name, cache)
            return getattr(instance, name)

        
        @wraps(fn)
        def func(self, *args, **kwargs):

            """Call function, record arguments"""
           
            cache = get_cache(self)
            argnum = len(args)
            argvarnames = varnames[1 : argnum + 1]
            kwargs0 = dict(zip(argvarnames, args))
            kwargs0.update(kwargs)
            # if funcname == 'set_strategy':
            #     import pdb
            #     pdb.set_trace()
            
            for arg in exclude:
                try:
                    kwargs0.pop(arg)
                except KeyError:
                    pass
                
            if replace:
                cache.replace_record(funcname, kwargs0)
            else:
                cache.append_record(funcname, kwargs0)
                
            return fn(self, *args, **kwargs)
        
        return func   
    
    
    return decorator


def flatten_record_actions(a):
    """Flatten the records list generated from `record_actions`
    
    Parameters
    -----------
    a : list
        Attribute created by decorator `record_actions`
    
    Returns
    --------
    out : dict
        Dictionary of record, hopefully easier to read.
        
        key is now 'number.funcname.argument'
    """
    newdict = {}
    for i, ai in enumerate(a):
        funcname = ai[0]
        adict = ai[1]
        
        # Record data even if method has no arguments.
        if len(adict) == 0:
            newkey = '%s.%s.' % (i, funcname)
            newdict[newkey] = None
        
        # Record all arguments of method
        for k, v in adict.items():
            newkey = '%s.%s.%s' % (i, funcname, k)
            newdict[newkey] = v
    return newdict
        
        
class RecordActionCache(object):
    """
    RecordActionCache records two kinds of method actions
    
    - Method calls that are recorded sequentially
    - Method calls that replace the input of previous calls
    
    Attributes
    ----------
    dict : dict
        Dictionary containing all recorded method actions and arguments. 
    
    """
    def __init__(self):
        self.reset()
        return
    
    def replace_record(self, funcname, kwargs):
        """Modify records that replace the previous method record"""
        
        if funcname not in self._replace:
            fnum = len(self._appended)
            self._replace[funcname] = fnum
            self.append_record(funcname, kwargs)
        else:
            fnum = self._replace[funcname]
            self._appended[fnum] = (funcname, kwargs)
        
        
    def append_record(self, funcname, kwargs):
        """Modify records that append to previous records"""
        record = (funcname, kwargs)
        self._appended.append(record)
        
        
    def reset(self):
        """Clear all records"""
        self._replace = {}
        self._appended = []
        return
        
        
    @property
    def dict(self):
        """dict of records.
        
        For appended, ordered method calls:
            
          - keys = callnum.funcname.kwarg
          - values = keyword argument values
          
        For replaced method calls:
            
         - keys = funcname.kwarg
         - value = keyword argument values
        """
        
        d = flatten_record_actions(self._appended)
        return d
    
    
    def repeat(self, obj):
        """Repeat the recorded method calls on the input object.
        
        Parameters
        -----------
        obj : object
            Parent object of stored methods
            
        """        
        for (funcname, kwargs) in self._appended:
            method = getattr(obj, funcname)
            method(**kwargs)   
        return
    
        
    
    
    @staticmethod
    def run_dict(d, obj):
        run_dict(d, obj)
    
    
def run_dict(d: dict, obj):
    """Take RecordActionCache.dict and convert it to RecordActionCache
    data
    
    Parameters
    ------------
    d : dict
        RecordActionCache.dict 
    obj : object
        Parent object to re-run method calls.     
    
    """
    d = misc.unflatten_dict(d, sep='.')
    d = filter_args(d)
    logger.debug('running record for %s, %s', obj, d)
    for key, value in d.items():
        
        ## Ordered append records start with numeric
        if key[0].isnumeric():
            keys = list(value.keys())
            values = list(value.values())
            funcname = keys[0]
            kwargs = values[0]
        else:
            funcname = key
            kwargs = value
        
        method = getattr(obj, funcname)
        
        ### method calls with no arguments have argument named ''
        ### Filter them out. 
        try:
            kwargs.pop('')
        except KeyError:
            pass
        method(**kwargs)
    return


def filter_args(d: dict) -> dict:
    """Filter arguments for invalid entries such as '' and nan.
    This is a sub-function for `run_dict`. """
    new = {}
    for key, value in d.items():
        
        keep = True
        
        if key == '':
            keep = False
            
        elif hasattr(value, 'items'):
            value = filter_args(value)

        else:
            try: 
                isnan = math.isnan(value)
                if isnan: 
                    keep = False
            except TypeError:
                pass
       
        if keep:
            new[key] = value
    return new
        



