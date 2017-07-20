#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Homing
@software: PyCharm Community Edition
@file: pandas_basic_function.py
@time: 2017/7/20 20:38
"""

import pandas as pd
import numpy as np


index = pd.date_range('1/1/2000', periods=8)
s = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
df = pd.DataFrame(np.random.randn(8, 3), index=index,columns=['A', 'B', 'C'])
wp = pd.Panel(np.random.randn(2, 5, 4), items=['Item1', 'Item2'],
major_axis=pd.date_range('1/1/2000', periods=5),minor_axis=['A', 'B', 'C', 'D'])
# 9.1 Head and Tail
long_series = pd.Series(np.random.randn(1000))
# print(long_series.head())
# print(long_series.tail(3))
# 9.2 Attributes and the raw ndarray(s)
"""
– Series: index (only axis)
– DataFrame: index (rows) and columns
– Panel: items, major_axis, and minor_axis

"""
# print(df[:2])
df.columns = [x.lower() for x in df.columns]
"""
print(df)
print(s.values)
print(df.values)
print(wp.values)
"""
# 9.3 Accelerated operations
# 9.4 Flexible binary operations
# 9.4.1 Matching / broadcasting behavior
df = pd.DataFrame({'one' : pd.Series(np.random.randn(3), index=['a', 'b', 'c']),
 'two' : pd.Series(np.random.randn(4), index=['a', 'b', 'c', 'd']),
'three' : pd.Series(np.random.randn(3), index=['b', 'c',
'd'])})
# print(df)
row = df.iloc[1]
column = df['two']
"""
print(df.sub(row, axis='columns'))
print(df.sub(row, axis=1))
"""
# 9.4.2 Missing data / operations with fill values
df2 = df.copy()
"""
print(df + df2)
print(df.add(df2, fill_value=0))
"""
# 9.4.3 Flexible Comparisons
"""
print(df.gt(df2))
print(df2.ne(df))
"""
# 9.4.4 Boolean Reductions
print((df > 0).all())
print((df > 0).any())
print(df.empty)
# 9.4.5 Comparing if objects are equivalent



