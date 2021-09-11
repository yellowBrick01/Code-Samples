# -*- coding: utf-8 -*-
"""
Created on Sun May  2 22:37:47 2021

@author: pdmuser
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from matplotlib.pyplot import show 
import matplotlib.image as mpimg

#Reading data from CSV file
data = pd.read_csv('clean_listings.csv')

#Creating a copy of dataframe
df = data.copy()
colors = ['#0818A8','#0437F2','#87CEEB','#B6D0E2', '#A7C7E7']
colorsv2 = ['#A7C7E7', '#B6D0E2', '#87CEEB', '#0437F2', '#0818A8']
plt.style.use('fivethirtyeight')
plt.figure(figsize=(13,7))
plt.title("Neighbourhood Group")
g = plt.pie(df.neighbourhood_group_cleansed.value_counts(),labels=df.neighbourhood_group_cleansed.value_counts().index,autopct='%1.1f%%', colors=colors,startangle=180)
plt.show()