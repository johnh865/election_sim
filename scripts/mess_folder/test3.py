# -*- coding: utf-8 -*-



def gen1():
    for i in range(100):
        yield i
        
        
for i in gen1(): print(i)
for i in gen1(): print(i)
a = gen1()
for i in a: print(i)