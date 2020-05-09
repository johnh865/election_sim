# -*- coding: utf-8 -*-
# import unittest

import votesim
import logging

import os
import time



def test_logger():
    filename = 'log-test.log'
    ls = votesim.logSettings
    ls.start_debug(filename=filename)
    print(ls.path)
    logger = logging.getLogger()

    s = 'This is the test message'
    logger.info(s)



    with open(ls.path, 'r') as f:
        s2 = f.read()

        
    assert s in s2
    
    print('\nPRINTING LOG CONTENTS')
    ls.print()
    ls.delete()
    return


    
if __name__ == '__main__':
    logging.shutdown()
    test_logger()
    

    
