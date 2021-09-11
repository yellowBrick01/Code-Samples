# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 22:31:22 2021

@author: pdmuser
"""

import pandas as pd

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', None)

#Read in the file: df1
airquality = pd.read_csv('airquality.csv')

print(airquality)

#Print the head of airquality
print(airquality.head())

#Melt airquality: airquality_melt
airquality_melt = pd.melt(airquality, id_vars=['Month','Day'])


#Print the head of airquality_melt
print(airquality_melt)

print(airquality, airquality_melt)

