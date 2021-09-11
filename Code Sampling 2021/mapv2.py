# -*- coding: utf-8 -*-
"""
Created on Sun May  2 22:02:02 2021

@author: pdmuser
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from matplotlib.pyplot import show 
import matplotlib.image as mpimg 
map_img = mpimg.imread('map.png') 

#Reading data from CSV file
data = pd.read_csv('clean_listings.csv')

#Creating a copy of dataframe
df = data.copy()
average_price = df["price"].mean()# To get the average price
high_listings = df[df['price'] > average_price] # Dataframe to hold the listings that are priced above
low_listings = df[df['price'] < average_price] # Dataframe to hold the listings that are priced below


#plt.figure(figsize=(10,6))
#map_nyc = sns.scatterplot(df.longitude,df.latitude,hue=df.neighbourhood_group_cleansed)
#map_nyc.imshow(map_img,
          #aspect = map_nyc.get_aspect(),
          #extent = map_nyc.get_xlim() + map_nyc.get_ylim(),
          #zorder = 0) 
#show()

plt.figure(figsize=(10,6))
#ny_state_m = plt.imread('map.png')
map_nyc = sns.scatterplot(high_listings.longitude,high_listings.latitude,hue=high_listings.neighbourhood_group_cleansed)
#map_nyc.fig.suptitle('Overall Title')
map_nyc.imshow(map_img,
          aspect = map_nyc.get_aspect(),
          extent = map_nyc.get_xlim() + map_nyc.get_ylim(),
          zorder = 0) 
show()