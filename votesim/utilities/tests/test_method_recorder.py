# -*- coding: utf-8 -*-
"""
Test if the recorder and record and output the right arguments
"""

import votesim
from votesim.utilities import recorder
from votesim.utilities.misc import flatten_dict, unflatten_dict


class MyObj(object):
    def __init__(self, x):
        self.arg_record = recorder.RecordActionCache()
        self.x = x
        self.z = 0
        
        
    @recorder.record_actions('arg_record')
    def square(self):
        self.x = self.x **2
        
        
    @recorder.record_actions('arg_record')
    def add(self, y, z=20):
        self.x = self.x + y + z
        
        
    @recorder.record_actions('arg_record')
    def add2(self, y):
        self.x = self.x + y
        
        
    @recorder.record_actions('arg_record', replace=True)
    def replace(self, z, b):
        self.z = z
        
        
t = MyObj(2)
t.square()
t.add(1, z=230)
t.add(10, 23)
t.add2(2)
t.replace(120, 50)
t.replace([1,2,3,4], {'a':10})


record = t.arg_record

t2 = MyObj(2)
recorder.run_dict(record.dict, t2)

f1 = flatten_dict(t.arg_record.dict)
f2 = flatten_dict(t2.arg_record.dict)


def test():
    for key in f1:
        assert f1[key] == f2[key]
    
    assert f1['0.square.'] == None
    assert f1['1.add.y'] == 1
    assert f1['1.add.z'] == 230
    assert f1['2.add.y'] == 10
    assert f1['2.add.z'] == 23
    assert f1['3.add2.y'] == 2
    assert f1['4.replace.z'] == 120
    assert f1['4.replace.b'] == 50
    
    
