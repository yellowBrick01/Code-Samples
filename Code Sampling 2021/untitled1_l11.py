# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 22:46:18 2021

@author: pdmuser
"""

import pandas as pd

#Read in the file: df1
tb = pd.read_csv('tb.csv')
print(tb)

tb_melt = pd.melt(tb, id_vars=['country','year'])
print(tb_melt)

#Print the head of tb_melt
print(tb_melt.head())

#Create the 'gender' column
tb_melt['gender'] = tb_melt.variable.str[0]
print(tb_melt['gender'].str[0])

pd.set_option('display.max_rows', None,'display.max_columns', None,'display.max_colwidth', None, 'display.width', None)

#Print the head of tb_melt
print(tb_melt.head())