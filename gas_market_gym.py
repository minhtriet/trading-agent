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

class GasMarket(gym.Env):
    "Simple marketEnv  Aim to teach: sense of urgency"

    def _str_to_date(self, s):
        month, year = s.split('/')
        if month[0] == 'M':
            return datetime.datetime(2000 + int(year), int(month[1:]), 1, 0, 0).date()
        else:
            pass

    def __init__(self, start_buying_date='1 3 2015', start_consuming_date='1 4 2015',
                 stop_consuming_date='1 5 2015', reservoir = 0):
        MAX_BUY_VOLUME = 1000
#        self.observation_space = spaces.Dict({
#            'future':  spaces.Dict({
#                'M1': spaces.Box(low=-100, high=100, shape=(3)),
#                'M2': spaces.Box(low=-1, high=1, shape=(3)),
#                'M3': spaces.Box(low=-1, high=1, shape=(3)),
#            }),
#            'is_gpl': spaces.Dict({
#            })
#        })
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
        "A set of 4 years * 2 summer * 2 winter * 5 quarter * 4 months * 2 market"
        done = False
        future, future_index, spot, demand = self._get_obs()

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
        if future != -1:
            for i, future in enumerate(gpl.columns):
                if self. _str_to_date(future) > self.stop_consuming_date:
                    return self._get_obs(), -10, True, {}
                else:
                    self.reservoir[self._str_to_date(future)] +=  [i]

        self.total_cost[self.today] += np.matmul(action, [gpl, ncg, spot])
        self.today = self.today + relativedelta(days = 1)

        return self._get_obs(), 0, done, {}

    def _get_obs(self):
        # future, sport, demand of today
        spot = self.spot_price[self.spot_price.Tradingday == self.today]
        demand = self.demand[self.demand.Zeitstempel == self.today]

        future_gpl = self.future_price_gpl[self.future_price_gpl.Tradingday == self.today]
        future_gpl = future_gpl[future_gpl != 'n.a.'].dropna(axis='columns').drop('Tradingday', axis=1)
        future_ncg = self.future_price_ncg[self.future_price_ncg.Tradingday == self.today]
        future_ncg = future_ncg[future_ncg != 'n.a.'].dropna(axis='columns').drop('Tradingday', axis=1)

        future = future_gpl.append(future_ncg).reset_index().astype(float)
        if len(future) == 0:
            return (-1, -1, spot, demand)

        future_index = future.idxmin(axis=0)
        future_price = future.min()
        return (future_price, future_index, spot, demand)

    def reset(self):
        return self._get_obs()
