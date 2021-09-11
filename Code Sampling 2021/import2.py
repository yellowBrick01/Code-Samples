# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 22:58:06 2021

@author: pdmuser
"""

import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_csv("temp.csv")

df.plot(color = 'red')

plt.title("Temperature Values")

plt.xlabel("")

plt.ylabel("Index")

plt.show()