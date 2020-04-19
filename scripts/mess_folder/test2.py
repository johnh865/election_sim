# -*- coding: utf-8 -*-

import numba
import time
import numpy as np

def test1():
    num = int(2e7)
    arr = np.zeros(num)
    for i in range(num):
        arr[i] =  i + 2
    return arr


@numba.jit()
def test2():
    num = int(2e7)
    arr = np.zeros(num)
    for i in range(num):
        arr[i] =  i + 2
    return arr


def test3():
    num = int(2e7)
    arr = np.zeros(num)
    arr2 = np.arange(num)
    for i in np.nditer(arr2):
        arr[i] =  i + 2
    return arr





a = time.time()
test1()
b = time.time()
print(b - a)


a = time.time()
test2()
b = time.time()
print(b - a)


#a = time.time()
#test3()
#b = time.time()
#print(b - a)