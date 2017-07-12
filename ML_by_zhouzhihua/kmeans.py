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


def dist(num1,num2):
    t = (num1-num2)**2
    return float(t**0.5)


def mean_dist(dataSet):
    d ={}
    """
    t1 = np.percentile(a,10)
    t2 = np.percentile(a, 30)
    t3 = np.percentile(a, 50)
    t4 = np.percentile(a, 90)
    
    """
    a = np.array(dataSet)

    l = random.sample(dataSet, 4)
    l.sort()
    t1 = l[0]
    t2 = l[1]
    t3 = l[2]
    t4 = l[3]
    d[t1] = []
    d[t2] = []
    d[t3] = []
    d[t4] = []

    sum = 0
   # print(t1,t2,t3,t4)
    for data in dataSet:
        m = []
        for t in [t1,t2,t3,t4]:
            m.append(dist(data,t))

        for ti in [t1,t2,t3,t4] :
            if dist(ti,data) == min(m):
                d[ti].append(data)
                sum += dist(ti,data)

    print('sum',sum)
    print('first',[t1,t2,t3,t4])

    t1 = np.array(d[t1]).mean()
    t2 = np.array(d[t2]).mean()
    t3 = np.array(d[t3]).mean()
    t4 = np.array(d[t4]).mean()
    d = {}
    d[t1] = []
    d[t2] = []
    d[t3] = []
    d[t4] = []

    sum1 = 0
    for data in dataSet:
        m = []
        for t in [t1,t2,t3,t4]:
            m.append(dist(data,t))

        for ti in [t1,t2,t3,t4] :
            if dist(ti,data) == min(m):
                d[ti].append(data)
                sum1 += dist(ti,data)
    # print([t1,t2,t3,t4])
    print('sum1',sum1)
    print('Second', [t1, t2, t3, t4])
    sorted(d.keys())
    return d.values()


def main():
    df = pd.read_excel('km.xlsx',sheetname='Sheet1')
    l2 = df['x2'].values.T
    # print(mean_dist(list(l2)))
    k = mean_dist(list(l2))
    m = list(k)
    l =[item for sublist in m for item in sublist]
    # print(l)
    t = pd.DataFrame(l)

    plt.scatter(range(len(t)),t)
    plt.show()

if __name__ == '__main__':
    main()

