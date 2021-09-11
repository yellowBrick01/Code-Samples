# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 23:30:43 2021

@author: pdmuser
"""

import pandas as pd

pd.set_option('display.max_rows', None,'display.max_columns', None,'display.max_colwidth', None, 'display.width', None)

ebola = pd.read_csv('ebola.csv')

#print(ebola)
print(ebola)

#Melt eblola: ebola_melt
ebola_melt = pd.melt(ebola, id_vars=['Date','Day'], var_name='status_country',value_name='counts')

#Print the head of ebola_melt
print(ebola_melt.head())

status_country = ebola_melt["status_country"]

ebola_tidy = pd.concat([ebola_melt, status_country], axis = 1)

#Print the shape of ebola_tidy
print(ebola_tidy.shape)

#Print the head of ebola_tidy
print(ebola_tidy.head())