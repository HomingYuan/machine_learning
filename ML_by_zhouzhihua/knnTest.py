# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 11:49:16 2017

@author: user
"""

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt


def dist(num1, num2):
    t = (num1 - num2) ** 2
    return float(t ** 0.5)


def Kmean(dataSet, k,oriPoint):
    d = {}
    for i in range(k):
        d[oriPoint[i]] = []
    sum = 0
    for i in range(k):
        for data in dataSet:
            m = [dist(oriPoint[i], data) for i in range(k)]
            if dist(data, oriPoint[i]) == min(m):
                d[oriPoint[i]].append(data)
                sum += min(m)

    for i in range(k):
        oriPoint[i] = np.array(d[oriPoint[i]]).mean()
    return oriPoint,d.values(),sum


def test(dataSet,k,n): # k 分的种类，n 循环次数
    point = []
    for i in range(n):
        point.append([])

    point[0]= sorted(random.sample(dataSet,k))
    t = 0
    while t < n-1:
        point[t+1] = Kmean(dataSet,k,point[t])[0]
        t += 1
    return Kmean(dataSet,k,point[n-1])[1],Kmean(dataSet,k,point[n-1])[2],point[n-1]


def main():
    df = pd.read_excel('km.xlsx', sheetname='Sheet1')
    l2 = df['x2'].values.T
    data_list = test(list(l2), 5, 30)
    k = data_list[0]
    Data = [item for sublist in k for item in sublist]
    print(data_list[2])
    print(data_list[1])
    plt.scatter(range(len(Data)), Data)
    plt.show()

if __name__ == "__main__":
    main()


