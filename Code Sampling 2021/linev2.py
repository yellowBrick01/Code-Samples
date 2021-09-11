# -*- coding: utf-8 -*-
"""
Created on Tue May  4 16:40:11 2021

@author: pdmuser
"""
#Import statements
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

#Reading data from CSV file
data = pd.read_csv('clean_listings.csv')

#Creating a copy of dataframe
df = data.copy()

#To create a new dataframe to hold the data that will be used for
#The line chart.
time_df = df.groupby('last_review_year').sum()[['number_of_reviews']]
#Uncomment 22 to see our new dataframe
#print(time_df)

#Lines 24 & 25 to convert the index from time_df and assign the output
#to a new dataframe called 'time_df_fin_ver'
time_df.reset_index(inplace=True)
time_df_fin_ver = time_df.rename(columns = {'index':'group_by_values'})
#Uncomment 29 to see the new data frame
#print(time_df_fin_ver)

#Lines 32 & 33 assign values to the variables that are going to store
# The year (x-axis) and total number of revie2s(y-axis)
y_axis = time_df_fin_ver['number_of_reviews']
x_axis = time_df_fin_ver['last_review_year']
#Ucomment Lines 36 & 37 to see the variables for our y and x axis
#print(y_axis)
#print(x_axis)

#Lines 40--46: Creates and define the paramaters for the line chart.
plt.plot(x_axis, y_axis, marker = 'o')
plt.xlabel("Years")
plt.ylabel("Number of reviews")
plt.title("The total number of reviews by year")
plt.xticks(x_axis)
plt.grid()
plt.show()