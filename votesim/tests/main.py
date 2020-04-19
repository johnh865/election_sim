# -*- coding: utf-8 -*-

#import election_sim
#from election_sim.utilities import detectfiles


import unittest
loader = unittest.TestLoader()
start_dir = ''
suite = loader.discover(start_dir)

runner = unittest.TextTestRunner()
runner.run(suite)

