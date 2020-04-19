# -*- coding: utf-8 -*-

import os
import fnmatch
import errno
import collections


def detectfiles(folder1, pattern):
    """Recursively detect files in path of folder1 using a pattern as 
    recognized by fnmatch"""
    matches = []
    for root, dirnames, filenames in os.walk(folder1):
      for filename in fnmatch.filter(filenames, pattern):
          matches.append(os.path.join(root, filename))
    return matches


def create_file_dirs(filename):
    """
    Construct directories for file recursively.
    
    From stackoverflow 
    https://stackoverflow.com/questions/12517451/automatically-creating-directories-with-file-output
    
    """
    
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    return


# def create_dirs(path):
#     """Create directories recursively"""
#     if not os.path.exists(path):
#         try:
#             os.makedirs(path)
#         except OSError as exc:
#             if exc.errno != errno.EEXIST:
#                 raise
#     return





def flatten_dict(d, parent_key='', sep='.'):
    """Flatten a nested dictionary of dictionaries.
    
    Parameters
    ----------
    d : dict
        Dictionary of dictionaries to flatten
    sep : str
        Symbol used to separate appended key names 
        
    Returns
    ---------
    out : dict
        Flattened dictionary where all sub-dictionaries are flattened into out.
    
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(dictionary, sep='.'):
    """Unflatten a dictionary and convert into nested dictionaries
    
    https://stackoverflow.com/questions/6037503/python-unflatten-dict
    
    Parameters
    -----------
    d : dict
        Dictionary 
    
    Returns
    --------
    out : dict
        Unflattened dictionary including sub-dictionaries are unflattened.
    
    """
    
    resultDict = dict()
    for key, value in dictionary.items():
        parts = key.split(sep)
        d = resultDict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return resultDict


    

