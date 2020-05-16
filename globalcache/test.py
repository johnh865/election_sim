# -*- coding: utf-8 -*-

import globalcache
import time


sleeptime = 1


@globalcache.cache_decorate('func1')
def func(x):
    time.sleep(sleeptime)
    return x + 4


print('global dict')
print(globals().keys())

g = globalcache.create(globals())


print(g)
t1 = time.time()
y1 = func(1)
t2 = time.time()
if (t2 - t1) < sleeptime:
    assert True
else:
    assert t2 - t1 > sleeptime
print('time #1 = ', t2 - t1)

t1 = time.time()
y2 = func(1)
t2 = time.time()
print('time #2 = ', t2 - t1)
assert (t2 - t1) < sleeptime
assert y1 == y2
