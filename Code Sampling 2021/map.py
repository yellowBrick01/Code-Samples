# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 20:12:08 2021

@author: pdmuser
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('clean_listings.csv')
#print(df.columns)
#print(df["price"].describe())
average_price = df["price"].mean()# To get the average price
high_listings = df[df['price'] > average_price] # Dataframe to hold the listings that are priced above
low_listings = df[df['price'] < average_price] # Dataframe to hold the listings that are priced below

#Please note that will we use the same boundary box for our map
BBox = ((df.longitude.min(),   df.longitude.max(), df.latitude.min(), df.latitude.max()))
#print(BBox)
#Please note will we use the same base image for the map
ny_state_m = plt.imread('map.png')


#Uncomment lines 27--32 for a map of a listings
#fig, ax = plt.subplots(figsize = (8,7))
#ax.scatter(df.longitude, df.latitude, zorder=1, alpha= 0.2, c='purple', s= 10)
#ax.set_title('Plotting Spatial Data on New York Map')
#ax.set_xlim(BBox[0],BBox[1])
#ax.set_ylim(BBox[2],BBox[3])
#ax.imshow(ny_state_m, zorder=0, extent = BBox, aspect= 'equal')

#Uncomment lines 35--40 for a map of listings that are priced above the average
#fig, ax = plt.subplots(figsize = (8,7))
#ax.scatter(high_listings.longitude, high_listings.latitude, zorder=1, alpha= 0.2, c='b', s=10)
#ax.set_title('Plotting locations of listings that are priced above the average.')
#ax.set_xlim(BBox[0],BBox[1])
#ax.set_ylim(BBox[2],BBox[3])
#ax.imshow(ny_state_m, zorder=0, extent = BBox, aspect= 'equal')

#Uncomment lines 43--48 for a map of listings that are priced below the average.
#fig, ax = plt.subplots(figsize = (8,7))
#ax.scatter(low_listings.longitude, low_listings.latitude, zorder=1, alpha= 0.2, c='r', s= 10)
#ax.set_title('Plotting the locations of listings that are priced below the average.')
#ax.set_xlim(BBox[0],BBox[1])
#ax.set_ylim(BBox[2],BBox[3])
#ax.imshow(ny_state_m, zorder=0, extent = BBox, aspect= 'equal')