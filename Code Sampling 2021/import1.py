# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 22:46:13 2021

@author: pdmuser
"""

import pandas as pd

#data frame structure

df = pd.read_csv("world_population.csv")

#print(df)

new_labels = ('Year of Population', 'World Popluation')

#integrate new labels
df2 = pd.read_csv("world_population.csv", header = 0, names = new_labels)

print(df2)