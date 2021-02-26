# -*- coding: utf-8 -*-
"""
Test special data that gave problems for irv voting
"""
import re
from ast import literal_eval

import numpy as np
import votesim
import votesim.votemethods.irv as irv

def test1():
    b = [[1, 0],
           [1, 0],
           [1, 0],
           [1, 0],
           [1, 0],
           [1, 0],
           [1, 0],
           [1, 0],
           [1, 0],
           [1, 0],
           [1, 0],
           [1, 0],
           [1, 0],
           [1, 0],
           [1, 0]]
    
    
    w, t, d = irv.irv(b)
    w2, t2, d2 = irv.irv_stv(b)
    
    assert 0 in w
    assert 0 in w2
    assert len(t) == 0
    assert len(t2) == 0
    assert len(w) == 1
    assert len(w2) == 1
    
    
def test2():
    b = [
     [0, 1],
     [0, 0],
     [0, 0],
     [0, 0],
     [0, 0],
     [0, 0],
     [0, 0],
     [0, 0],
     [0, 1],
     [0, 0]]
    
    w, t, d = irv.irv(b)
    w2, t2, d2 = irv.irv_stv(b)
    assert 1 in w
    assert 1 in w2