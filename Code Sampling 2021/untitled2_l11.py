# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 23:00:15 2021

@author: pdmuser
"""

import pandas as pd

#Read in the file: df1
ebola = pd.read_csv('ebola.csv')
print(ebola.head())

#Melt eblola: ebola_melt
ebola_melt = pd.melt(ebola, id_vars=['Date','Day'], var_name='type_country',value_name='counts')

#Print the head of ebola_melt
print(ebola_melt.head())

#Create the 'str_split' column
ebola_melt['str_split'] = ebola_melt.type_country.str.split('_')

#print(ebola_melt['str_split'])

# Create the 'type' column
ebola_melt['type']= ebola_melt.str_split.str.get(0)

# Create the 'country' column
ebola_melt['country'] = ebola_melt.str_split.str.get(1)

print(ebola_melt.head())