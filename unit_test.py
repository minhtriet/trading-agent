#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 00:10:02 2018

@author: mtriet
"""

import unittest
from gas_market_gym import GasMarket

class TestStringMethods(unittest.TestCase):

    def setUp(self):
        self.market = GasMarket()

    def test_buy(self):
        market = self.market
        state, reward, done, _ = market.step([100] + [0]*4 + [0]*4)
        assertEqual(market.reservoir[])

    
if __name__ == '__main__':
    unittest.main()