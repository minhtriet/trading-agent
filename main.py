# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 19:09:08 2018

@author: GIQNO
"""

import datetime
import numpy as np

from gas_market_gym import GasMarket

start_buying_date = '1 12 2014'

start_consuming_date = '1 1 2015'

end_consuming_date = '2 1 2015'

gm = GasMarket(start_buying_date, start_consuming_date, end_consuming_date)
# action: which month of which market

start_buying_date = datetime.datetime.strptime(start_buying_date, '%d %m %Y')
end_consuming_date = datetime.datetime.strptime(end_consuming_date, '%d %m %Y')

for _ in range((end_consuming_date - start_buying_date).days):
    action = np.random.randint(0, gm.MAX_BUY_VOLUME, 9) 
    gm.step(action)
    