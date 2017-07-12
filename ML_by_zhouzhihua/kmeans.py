# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 11:49:16 2017

@author: user
"""

import pandas as pd
import math
import numpy as np
import random
import matplotlib.pyplot as plt
"""
df=pd.read_excel("km.xlsx",sheetname="Sheet1")
# l1=df['x1'].values.T
l2=df['x2'].values.T
random_num = np.random.choice(l2,4)
print(random_num)

"""
def dist(num1,num2):
    t = (num1-num2)**2
    return float(t**0.5)


def mean_dist(dataSet):
    d ={}
    a = np.array(dataSet)
    t1 = np.percentile(a,20)
    t2 = np.percentile(a, 40)
    t3 = np.percentile(a, 60)
    t4 = np.percentile(a,80)
    d[t1] = []
    d[t2] = []
    d[t3] = []
    d[t4] = []
    distances = []
   # print(t1,t2,t3,t4)
    for data in dataSet:
        m = []
        for t in [t1,t2,t3,t4]:
            m.append(dist(data,t))
        # distances.append(min(m))
        if dist(t1,data) == min(m):
            d[t1].append(data)
        if dist(t2,data) == min(m):
            d[t2].append(data)
        if dist(t3,data) == min(m):
            d[t3].append(data)
        if dist(t4,data) == min(m):
            d[t4].append(data)
    sorted(d.keys())
    return d.values()

df = pd.read_excel('km.xlsx',sheetname='Sheet1')
l2 = df['x2'].values.T
# print(mean_dist(list(l2)))
k = mean_dist(list(l2))
m = list(k)
l =[item for sublist in m for item in sublist]
# print(l)
t = pd.DataFrame(l)
print(t)
plt.scatter(range(len(t)),t)
plt.show()
