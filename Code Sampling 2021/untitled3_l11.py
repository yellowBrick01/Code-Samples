# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 23:15:35 2021

@author: pdmuser
"""

import pandas as pd

uber1 = pd.read_csv('nyc_uber_april.csv')
uber2 = pd.read_csv('nyc_uber_may.csv')
uber3 = pd.read_csv('nyc_uber_june.csv')

#Concatenate dataframes: uber1, uber2, uber3
row_concat = pd.concat([uber1, uber2, uber3])

#Print the shape of row_concat
print(row_concat.shape)

row_concat.dropna()

pd.set_option('display.max_rows', None,'display.max_columns', None,'display.max_colwidth', None, 'display.width', None)

#Print the head of row_concat
print(row_concat.head())

print(row_concat)
