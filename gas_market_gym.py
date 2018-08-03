# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 14:47:07 2018

@author: GIQNO
"""

import gym
from gym import spaces
from gym.utils import seeding
import pandas as pd

from dateutil.relativedelta import relativedelta

import datetime
import numpy as np

import util


def _str_to_date(s):
    # TODO quarter, years, seasons
    month, year = s.split('/')
    if month[0] == 'M':
        return datetime.datetime(2000 + int(year), int(month[1:]), 1, 0, 0).date()
    else:
        pass

class GasMarket(gym.Env):
    """Simple marketEnv
       Aim to teach: sense of urgency
    """
    
    def __init__(self, start_buying_date, start_consuming_date, stop_consuming_date, reservoir = 0): 
        self.MAX_BUY_VOLUME = 1000
        self.observation_space = spaces.Tuple([spaces.Box(low = np.array([0]), high = np.array([MAX_BUY_VOLUME]), dtype=np.float32),
                                         spaces.Box(low = np.array([0]), high = np.array([MAX_BUY_VOLUME]), dtype=np.float32),
                                         spaces.Box(low = np.array([0]), high = np.array([MAX_BUY_VOLUME]), dtype=np.float32),
                                         spaces.Box(low = np.array([0]), high = np.array([MAX_BUY_VOLUME]), dtype=np.float32),
                                         spaces.Box(low = np.array([0]), high = np.array([MAX_BUY_VOLUME]), dtype=np.float32),
                                         spaces.Box(low = np.array([0]), high = np.array([MAX_BUY_VOLUME]), dtype=np.float32),
                                         spaces.Box(low = np.array([0]), high = np.array([MAX_BUY_VOLUME]), dtype=np.float32),
                                         spaces.Box(low = np.array([0]), high = np.array([MAX_BUY_VOLUME]), dtype=np.float32),
                                         spaces.Box(low = np.array([0]), high = np.array([MAX_BUY_VOLUME]), dtype=np.float32)])
        # future, spot, demand of today
#        self.action_space = spaces.Tuple(spaces.Box(low = np.array([0, 0]), high = np.array([MAX_BUY_VOLUME, MAX_BUY_VOLUME]), dtype=np.float32), spaces.Discrete(4), spaces.Discrete(2), spaces.Discrete(2),
#                                        spaces.Discrete( 2 + 4 + 4))  # Y + S + W + Q + M
        self.action_space = spaces.Tuple([spaces.Box(low = np.array([0]), high = np.array([MAX_BUY_VOLUME]), dtype=np.float32),
                                         spaces.Box(low = np.array([0]), high = np.array([MAX_BUY_VOLUME]), dtype=np.float32),
                                         spaces.Box(low = np.array([0]), high = np.array([MAX_BUY_VOLUME]), dtype=np.float32),
                                         spaces.Box(low = np.array([0]), high = np.array([MAX_BUY_VOLUME]), dtype=np.float32),
                                         spaces.Box(low = np.array([0]), high = np.array([MAX_BUY_VOLUME]), dtype=np.float32),
                                         spaces.Box(low = np.array([0]), high = np.array([MAX_BUY_VOLUME]), dtype=np.float32),
                                         spaces.Box(low = np.array([0]), high = np.array([MAX_BUY_VOLUME]), dtype=np.float32),
                                         spaces.Box(low = np.array([0]), high = np.array([MAX_BUY_VOLUME]), dtype=np.float32),
                                         spaces.Box(low = np.array([0]), high = np.array([MAX_BUY_VOLUME]), dtype=np.float32),])
        # read data  
        assert reservoir >= 0
        start_buying_date = datetime.datetime.strptime(start_buying_date, '%d %m %Y').date()
        start_consuming_date = datetime.datetime.strptime(start_consuming_date, '%d %m %Y').date()
        stop_consuming_date = datetime.datetime.strptime(stop_consuming_date, '%d %m %Y').date()
        assert stop_consuming_date > start_consuming_date

        # reservoir
        dates = pd.date_range(start=start_buying_date, end=stop_consuming_date).date
        self.reservoir = dict(zip(dates, [0]*len(dates)))
        self.total_cost = dict(zip(dates, [0]*len(dates)))
        
        self.demand = util.read_demand()
        self.spot_price = util.read_spot_market()
        self.future_price_gpl, self.future_price_ncg = util.read_future_market()        
        self.start_buying_date = start_buying_date
        self.start_consuming_date = start_consuming_date
        self.today = self.start_buying_date
        self.stop_consuming_date = stop_consuming_date
        self.seed()
        self.reset()

    def seed(self, seed=47):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    

    def step(self, action):
        """
        A set of 4 years * 2 summer * 2 winter * 5 quarter * 4 months * 2 market
        Currently 4 months * 2 market
        """
        done = False
        gpl, ncg, spot, demand = self._get_obs()        
        gpl = gpl[gpl.columns[gpl.columns.str.startswith('M')]].dropna(axis='columns')
        ncg = ncg[ncg.columns[ncg.columns.str.startswith('M')]].dropna(axis='columns')
        
        # minus reservoir
        if self.today >= self.start_consuming_date:
            self.reservoir[self.today] -= demand
        # stop condition
        if self.reservoir[self.today] < 0:
            reward = -10
            done = True
            return self._get_obs(), reward, done, {} 
        if self.today == self.stop_consuming_date:
            # some sense of preparing for the next period?
            # TODO buy in far future, passing the stop consuming date, is not rewarded 
            reward = np.mean(self.spot_price) - self.total_cost[self.today]
            reward += self.reservoir[self.today] * spot
            done = True
            return self._get_obs(), reward, done, {}
        # buying
        self.reservoir[self.today + relativedelta(days = 1)] += action[-1]  # spot_market
        for i, future in enumerate(gpl.columns):
            if _str_to_date(future) > self.stop_consuming_date:
                return self._get_obs(), -10, True, {}
            else:
                self.reservoir[_str_to_date(future)] += action[i]
        
        self.total_cost[self.today] += np.matmul(action, [gpl, ncg, spot])
        self.today = self.today + relativedelta(days = 1)

        return self._get_obs(), 0, done, {}

    def _get_obs(self):
        # future, sport, demand of today
        future_gpl = self.future_price_gpl[self.future_price_gpl.Tradingday == self.today]
        future_gpl = future_gpl[future_gpl != 'n.a.'].dropna(axis='columns').drop('Tradingday', axis=1)      
        future_ncg = self.future_price_ncg[self.future_price_ncg.Tradingday == self.today]
        future_ncg = future_ncg[future_ncg != 'n.a.'].dropna(axis='columns')        
        spot = self.spot_price[self.spot_price.Tradingday == self.today]        
        demand = self.demand[self.demand.Zeitstempel == self.today]
        return (future_gpl, future_ncg, spot, demand)

    def reset(self):
        return self._get_obs()
