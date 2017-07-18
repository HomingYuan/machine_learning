#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Homing
@software: PyCharm Community Edition
@file: pandas_cookbook.py
@time: 2017/7/18 20:53
"""
import pandas as pd
import numpy as np

# 7.1 Idioms
df = pd.DataFrame({'AAA' : [4,5,6,7], 'BBB' : [10,20,30,40],'CCC' : [100,50,-30,-50]})
# print(df)
# 7.1.1 if-then...
df.loc[df.AAA >= 5,'BBB'] = -1
# print(df)
df.loc[df.AAA >= 5,['BBB','CCC']] = 555
# print(df)
df.loc[df.AAA < 5,['BBB','CCC']] = 2000
# print(df)
df_mask = pd.DataFrame({'AAA' : [True] * 4, 'BBB' : [False] * 4,'CCC' : [True, False] * 2})
# print(df.where(df_mask,-1000))
# 7.1.2 Splitting
dflow = df[df.AAA <= 5]
# print(dflow)
dfhigh = df[df.AAA > 5]
# print(dfhigh)

# 7.1.3 Building Criteria
newseries = df.loc[(df['BBB'] < 25) & (df['CCC'] >= -40), 'AAA']
print(newseries)







