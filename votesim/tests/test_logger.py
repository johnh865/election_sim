# -*- coding: utf-8 -*-
import unittest

import votesim
import logging


class TestIRV(unittest.TestCase):
    def test_logger(self):

        logger = logging.getLogger()
        votesim.log2file()
        logger.info('test log')
