# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 14:47:07 2018

@author: GIQNO
"""

import gym
from gym import spaces
from gym.utils import seeding
from datetime import date
import panda

MAX_BUY_VOLUME = 1000

def cmp(a, b):
    return float(a > b) - float(a < b)

class GasMarket(gym.Env):
    """Simple marketEnv


    """
    def __init__(self, year):
        self.action_space = spaces.Box(low = 0, high = MAX_BUY_VOLUME)
        self.demand = (date(year+1,1,1) - date(year,1,1)).days * 24
        demand
        self.observation_space = spaces.Tuple(spaces.Discrete(d))
        self.seed()
        self.reset()

    def seed(self, seed=47):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        if action:  # hit: add a card to players hand and return
            self.player.append(draw_card(self.np_random))
            if is_bust(self.player):
                done = True
                reward = -1
            else:
                done = False
                reward = 0
        else:  # stick: play out the dealers hand, and score
            done = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(draw_card(self.np_random))
            reward = cmp(score(self.player), score(self.dealer))
            if self.natural and is_natural(self.player) and reward == 1:
                reward = 1.5
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player))

    def reset(self, year):
        self.dealer = draw_hand(self.np_random)
        self.player = draw_hand(self.np_random)
        return self._get_obs()
