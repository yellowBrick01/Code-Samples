# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 22:32:59 2021

@author: pdmuser
"""

import pandas as pd

df = pd.read_csv('car.csv')

print (df ['distance traveled'].min())

print (df ['distance traveled'].max())

#Summary Statistics
print (df ['distance traveled'].describe())