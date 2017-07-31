#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Homing
@software: PyCharm Community Edition
@file: pandas_cookbook.py
@time: 2017/7/28 09:16
"""

import pandas as pd
import numpy as np

df = pd.read_csv('job_analyse_zl.csv',encoding='utf-8',sep=',',header=None,names=['职位','公司','薪水','地区','发布时间'])
# print(df['发布时间'])
low_slary = []
high_slary = []
for i in df['薪水']:
    d = str(i)
    t = d.find('-')
    if t >= 0  :
        low_slary.append(d[0:t])
        high_slary.append(d[t+1:])
    elif t == -1 or ('北京' in d):
        low_slary.append(0)
        high_slary.append(0)
# print(high_slary)

"""
print(df['薪水'][18160:18180])
print(df['low'][18160:18180])
print(df['high'][18160:18180])
"""

remove_list = []
d  = list('0123456789')
for i in range(len(low_slary)):
    for m in d:
        if m in str(low_slary[i]):
            pass
        remove_list.append(0)


for item in remove_list:
    low_slary[item] = 0
    high_slary[item] = 0




df['low'] = low_slary
df['high']= high_slary
# df[['low','high']] = df[['low','high']].astype(float)
# print(df.dtypes)
# print(df['high'])
df.to_csv('data_process.csv',sep=',')