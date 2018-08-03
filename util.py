# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 18:05:28 2018

@author: GIQNO
"""

def root_msqe(y_test, y_score):
    return sum(y_test - y_score)**2

import pandas as pd

import os

demands_files = ['input/Weimar_Lastgang 2015_Erdgasbezug.xlsx', 
                 'input/Weimar_Lastgang 2016_Erdgasbezug.xlsx', 
                 'input/Kiel_Lastgang_KVP_KJ2015_07062016.xlsx']

def read_spot_market():
    try:
        spot = pd.read_hdf('input/Mappe1.h5', key = 'spot')
    except:
        xls = pd.read_excel('input/Mappe1.xlsx', sheet_name=None)
        xls['G_EEX_TRP'] = xls['G_EEX_TRP'].drop('Unnamed: 5', axis=1).iloc[1:]
        xls['G_EEX_TRP'].columns = xls['G_EEX_TRP'].iloc[0]
        spot = xls['G_EEX_TRP'].rename(columns={'Tradingday\nHandelstag': 'Tradingday'})
        spot = spot.iloc[1:]
        spot.Tradingday = spot.Tradingday.map(lambda x: x.date()) 
        spot.to_hdf('input/Mappe1.h5', key = 'spot')
    return spot

def read_future_market():
    #        TODO add year, season, quarter
    try:
        gpl = pd.read_hdf('input/Mappe1.h5', key = 'gpl')
        ncg = pd.read_hdf('input/Mappe1.h5', key = 'ncg')
    except:
        xls = pd.read_excel('input/Mappe1.xlsx', sheet_name=None)        
        xls['G_EEX_GPL'].columns = xls['G_EEX_GPL'].iloc[1]
        xls['G_EEX_GPL'] = xls['G_EEX_GPL'].iloc[2:]
        gpl = xls['G_EEX_GPL'].rename(columns={'Tradingday\nHandelstag': 'Tradingday'})
        gpl.Tradingday = gpl.Tradingday.map(lambda x: x.date())

        xls['G_EEX_NCG'].columns = xls['G_EEX_NCG'].iloc[1]
        xls['G_EEX_NCG'] = xls['G_EEX_NCG'].iloc[2:]
        ncg = xls['G_EEX_NCG'].rename(columns={'Tradingday\nHandelstag': 'Tradingday'})
        ncg.Tradingday = ncg.Tradingday.map(lambda x: x.date())

        gpl.to_hdf('input/Mappe1.h5', key = 'gpl')
        ncg.to_hdf('input/Mappe1.h5', key = 'ncg')
    return gpl, ncg

def read_demand():
    if not os.path.exists('input/demand.h5'):
        f = demands_files[0]
        xls = pd.read_excel(f, sheet_name=None)
        for sheet in xls:
            xls[sheet] = xls[sheet][['Unnamed: 16', 'Unnamed: 17']]    
            xls[sheet].columns = xls[sheet].iloc[1]
            xls[sheet] = xls[sheet].iloc[2:]
        demand = pd.concat(xls.values(), sort=False, ignore_index=True)
        demand = demand.groupby(demand.Zeitstempel.dt.date).sum().reset_index()
        demand.to_hdf('input/demand.h5', key = 'demand')
    else:
        demand = pd.read_hdf('input/demand.h5', key = 'demand')
    return demand 