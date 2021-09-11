# -*- coding: utf-8 -*-
"""
Created on Tue May  4 16:40:11 2021

@author: pdmuser
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
#Reading data from CSV file
data = pd.read_csv('clean_listings.csv')
#Creating a copy of dataframe
df = data.copy()
#print(df)
#print(df['last_review_year'])
#print(df['price'])
#x = df['last_review_year']
#print(x)
#y = df.groupby('last_review_year').count()[['price']]
#print(y)
x = df.last_review_year.unique()
print(x)
y = df.groupby('last_review_year').count().price.unique()
print(y)
print(y)
#print(y.columns)
#print(y.info())

#print(df['last_review_year])

plt.plot(x, y)
plt.show()

#print(df.columns)


