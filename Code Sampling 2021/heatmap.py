# -*- coding: utf-8 -*-
"""
Created on Thu May  6 13:40:17 2021

@author: pdmuser
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

#Reading data from CSV file
data = pd.read_csv('clean_listings.csv')

#Creating a copy of dataframe
df = data.copy()
#print(df.columns)
#print(df.head(10))

#Lines 22: Creates a new dataframe to group the average price of a listing by NYC borough and year
hm_df = df.groupby(['neighbourhood_group_cleansed','first_review_year']).mean()[['price']]
#print(hm_df.head(10))

#Line 26 & 27: To reset the index of hd_df and assign the value into a new data frame
hm_df.reset_index(inplace=True)
hm_df_v2 = hm_df.rename(columns = {'index':'group_by_values'})
#print(hm_df_v2.head(10))

#Line 31: Sort the results by price (High --> Low)
hm_df_v3 = hm_df_v2.sort_values('price',ascending = False)
#print (hm_df_v3.head(10))

#Line 35: Pivot value of hm_df_v3 and assign it to a new data frame
hm_df_finver= hm_df_v3.pivot('neighbourhood_group_cleansed','first_review_year','price')
print(hm_df_finver.head(10))

#Line 40--42: Creates the heatmap to display the average price of all listings 
#within a certain NYC borugh and Year 
ax = sns.heatmap(hm_df_finver)
plt.title("Heatmap: Average price by borough and year")
plt.show()

