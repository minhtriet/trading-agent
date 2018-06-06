# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 19:09:08 2018

@author: GIQNO
"""

import pandas as pd

import config

def preprocess():
    ncg = pd.read_excel('Prices_DB 201805025.xlsx', sheet_name='G_EEX_NCG')
    gpl = pd.read_excel('Prices_DB 201805025.xlsx', sheet_name='G_EEX_GPL')
    trp = pd.read_excel('Prices_DB 201805025.xlsx', sheet_name='G_EEX_TRP')
    ncg.to_csv('ncg.csv')
    trp.to_csv('trp.csv')
    