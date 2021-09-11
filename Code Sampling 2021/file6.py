# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 22:41:15 2021

@author: pdmuser
"""

import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_csv('titanic.csv')

print(df)

print (df['Fare'].describe())

#print (df['Fare'].mean())

#print (df['Fare'].std())

df['Fare'].plot(kind = 'box')

plt.show()