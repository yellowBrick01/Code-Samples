# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 22:52:30 2021

@author: pdmuser
"""

import pandas as pd

# Read in the file: df1
df1 = pd.read_csv('file_messy.csv')

print(df1.head())

df2 = pd.read_csv('file_messy.csv', delimiter= ' ', header= None)

print(df2.head())

#df2.to_csv('file_clean.csv', index = False)

df2.to_excel('file_clean1.xlsx', index = False)