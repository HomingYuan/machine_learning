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


def cenPoint(dataSet,k):
    oriPoint = random.sample(dataSet, k)
    oriPoint = sorted(oriPoint)
    print('first',oriPoint)
    d = {}
    for i in range(k):
        d[oriPoint[i]] =[]
    sum = 0
    for i in range(k):
        for data in dataSet:
            m = [dist(oriPoint[i],data) for i in range(k)]
            if dist(data,oriPoint[i]) == min(m):
                d[oriPoint[i]].append(data)
                sum += min(m)

    for i in range(k):
        oriPoint[i] = np.array(d[oriPoint[i]]).mean()

    d1= {}
    sum1 =0
    for i in range(k):
        d1[oriPoint[i]] = []

    for i in range(k):
        for data in dataSet:
            m = [dist(oriPoint[i],data) for i in range(k)]
            if dist(data,oriPoint[i]) == min(m):
                d1[oriPoint[i]].append(data)
                sum1 += min(m)

    for i in range(k):
        oriPoint[i] = np.array(d1[oriPoint[i]]).mean()

    d2= {}
    sum2 = 0
    for i in range(k):
        d2[oriPoint[i]] = []
    for i in range(k):
        for data in dataSet:
            m = [dist(oriPoint[i],data) for i in range(k)]
            if dist(data,oriPoint[i]) == min(m):
                d2[oriPoint[i]].append(data)
                sum2 += min(m)

    for i in range(k):
        oriPoint[i] = np.array(d2[oriPoint[i]]).mean()
    d3= {}
    sum3 = 0
    for i in range(k):
        d3[oriPoint[i]] = []
    for i in range(k):
        for data in dataSet:
            m = [dist(oriPoint[i],data) for i in range(k)]
            if dist(data,oriPoint[i]) == min(m):
                d3[oriPoint[i]].append(data)
                sum3 += min(m)

    for i in range(k):
        oriPoint[i] = np.array(d3[oriPoint[i]]).mean()
    d4= {}
    sum4 = 0
    for i in range(k):
        d4[oriPoint[i]] = []
    for i in range(k):
        for data in dataSet:
            m = [dist(oriPoint[i],data) for i in range(k)]
            if dist(data,oriPoint[i]) == min(m):
                d4[oriPoint[i]].append(data)
                sum4 += min(m)
    print('last',oriPoint)
    print(sum4)
    return d4.values()


def main():
    df = pd.read_excel('km.xlsx', sheetname='Sheet1')
    l2 = df['x2'].values.T
    for i in range(2):
        k = cenPoint(list(l2),4)
        #print(t)
        test = [item for sublist in k for item in sublist]
        plt.scatter(range(len(test)),test)
        plt.show()

if __name__ == "__main__":
    main()

