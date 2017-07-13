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


def dist(num1, num2):
    t = (num1 - num2) ** 2
    return float(t ** 0.5)


def mean_dist(dataSet):
    d = {}
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

    for data in dataSet:
        m = []
        for t in [t1, t2, t3, t4]:
            m.append(dist(data, t))

        for ti in [t1, t2, t3, t4]:
            if dist(ti, data) == min(m):
                d[ti].append(data)
                sum += dist(ti, data)
    sorted(d.keys())
    print(t1,t2,t3,t4)
    print(sum)
    return d

def classifier(dataSet):

    valDict = mean_dist(dataSet)
    key = valDict.keys()
    d = {}
    m1 = list(key)[0]
    m2 = list(key)[1]
    m3 = list(key)[2]
    m4 = list(key)[3]
    t1 = np.array(valDict[m1]).mean()
    t2 = np.array(valDict[m2]).mean()
    t3 = np.array(valDict[m3]).mean()
    t4 = np.array(valDict[m4]).mean()
    return mean_dist(dataSet)



def main():
    df = pd.read_excel('km.xlsx', sheetname='Sheet1')
    l2 = df['x2'].values.T
    for i in range(50):
        t = classifier(list(l2))
        #print(t)

        k = t.values()
        test = [item for sublist in k for item in sublist]
        plt.scatter(range(len(test)),test)
        plt.show()


if __name__ == "__main__":
    main()
