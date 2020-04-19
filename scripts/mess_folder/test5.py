# -*- coding: utf-8 -*-
from numba import jit
import numpy as np
import time

@jit
def zero_out_random(ratings, limits, weights=None, rs=None):
    
    ratings = np.copy(ratings)
    limits = limits.astype(int)
    vnum, cnum = ratings.shape
    remove_arr = np.maximum(0, cnum - limits)    
    
    if rs is None:
        rs = np.random.RandomState()
    
    for i, remove_num in enumerate(remove_arr):
        index = rs.choice(cnum, size=remove_num, p=weights, replace=False)
        ratings[i, index] = 0
    return ratings



    

def zero_out_random2(ratings, limits, weights=None, rs=None):
    ratings = np.copy(ratings)
    limits = limits.astype(int)
    vnum, cnum = ratings.shape
    remove_arr = np.maximum(0, cnum - limits)    
    
    if rs is None:
        rs = np.random.RandomState()
    
    for i, remove_num in enumerate(remove_arr):
        index = rs.choice(cnum, size=remove_num, p=weights, replace=False)
        ratings[i, index] = 0
    return ratings
        
    

num = 500000
ratings = np.random.rand(num, 10)
limits = np.random.randint(0, 10, size=num)
rs = np.random.RandomState(0)

t1 = time.time()
a = zero_out_random(ratings, limits, rs=rs)
t2 = time.time()
print(t2-t1)

t1 = time.time()
b = zero_out_random2(ratings, limits, rs=rs)
t2 = time.time()
print(t2-t1)
