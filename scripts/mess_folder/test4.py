# -*- coding: utf-8 -*-
import numpy as np
import votesim.votesystems.irv as irv
import pandas as pd


class Generator(object):
    def __init__(self, x):
        self.x = x
        self.generator = self.run()
        return
    
        
    def __next__(self):
        return next(self.generator)
    
    def __iter__(self):
        return self
    
    
    def run(self):
        for i in range(self.x):
            yield i
            
            
            
class Generator2(object):
    def __init__(self, x):
        self.x = x
        return
    
    def __call__(self, n):
        for i in range(n):
            yield self.x + i
            
            
            
g = Generator2(10)

for i in g(5): print(i)