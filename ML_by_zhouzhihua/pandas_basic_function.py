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
# print((df > 0).all())
# print((df > 0).any())
# print(df.empty)
# 9.4.5 Comparing if objects are equivalent
"""
print(df+df == df*2)
print(np.nan == np.nan)
print((df+df).equals(df*2))
"""
# 9.4.6 Comparing array-like objects
"""
print(pd.Series(['foo', 'bar', 'baz']) == 'foo')
print(pd.Index(['foo', 'bar', 'baz']) == 'foo')
print(pd.Series(['foo', 'bar', 'baz']) == pd.Index(['foo', 'bar', 'qux']))
print(pd.Series(['foo', 'bar', 'baz']) == np.array(['foo', 'bar', 'qux']))
print(pd.Series(['foo', 'bar', 'baz']) == pd.Series(['foo', 'bar'])) # Series lengths must match to compare
"""
# 9.4.7 Combining overlapping data sets
df1 = pd.DataFrame({'A' : [1., np.nan, 3., 5., np.nan], 'B' : [np.nan, 2., 3., np.nan, 6.]})
df2 = pd.DataFrame({'A' : [5., 2., 4., np.nan, 3., 7.], 'B' : [np.nan, np.nan, 3., 4., 6., 8.]})
"""
print(df1)
print(df2)
print(df1.combine_first(df2)) # if df1 item exist keep it else replace it by df2 item
print(df2.combine_first(df1)) # if df2 item exist keep it else replace it by df1 item
"""
# 9.5 Descriptive statistics
"""
print(df)
print(df.mean(0)) # statistics on index direction
print(df.mean(1)) # statistics on column direction
print(df.sum(0, skipna=False))
print(df.sum(0, skipna=True))
print(df.sum(axis=1, skipna=True))
print(df.sum(axis=1, skipna=False))
"""
ts_stand = (df - df.mean()) / df.std()
# print(ts_stand.std())
xs_stand = df.sub(df.mean(1), axis=0).div(df.std(1), axis=0)
# print(xs_stand.std(1))
# print(df)
# print(df.cumsum())
# print(np.mean(df['one']))
# print(np.mean(df['one'].values))
series = pd.Series(np.random.randn(500))
series[20:500] = np.nan
series[10:20] = 5
# print(series.nunique())
# 9.5.1 Summarizing data: describe
series = pd.Series(np.random.randn(1000))
series[::2] = np.nan
# print(series.describe())
frame = pd.DataFrame(np.random.randn(1000, 5), columns=['a', 'b', 'c', 'd','e'])
frame.iloc[::2] = np.nan # iloc refer to index
# print(frame.describe())
# print(series.describe(percentiles=[.05, .25, .75, .95]))
s = pd.Series(['a', 'a', 'b', 'b', 'a', 'a', np.nan, 'c', 'd', 'a'])
#print(s.describe())
frame = pd.DataFrame({'a': ['Yes', 'Yes', 'No', 'No'], 'b': range(4)})
"""
print(frame.describe())
print(frame.describe(include=['object']))
print(frame.describe(include=['number']))
print(frame.describe(include='all'))
"""
# 9.5.2 Index of Min/Max Values
"""
The idxmin() and idxmax() functions on Series and DataFrame compute the index labels with the minimum and
maximum corresponding values:
"""
s1 = pd.Series(np.random.randn(5))
# print(s1)
# print(s1.idxmin(),s1.idxmax())
df1 = pd.DataFrame(np.random.randn(5,3), columns=['A','B','C'])
# print(df1.idxmin(axis=0))
# print(df1.idxmin(axis=1))
# print(df1.idxmax(axis=0))
# print(df1.idxmax(axis=1))
df3 = pd.DataFrame([4, 1, 8, 3, np.nan], columns=['A'], index=list('edcba'))
# print(df3['A'].idxmin())
# print(df3['A'].idxmax()) # the index of max value(if have more than 1 item return first one)
# 9.5.3 Value counts (histogramming) / Mode
data = np.random.randint(0, 7, size=50)
s = pd.Series(data)
# print(s.value_counts())
# print(pd.value_counts(data)) # same as s.value_counts()
s5 = pd.Series([1, 1, 3, 3, 3, 5, 5, 7, 7, 7])
# print(s5.mode())
df5 = pd.DataFrame({"A": np.random.randint(0, 7, size=50), "B": np.random.randint(-10, 15, size=50)})
# print(df5.mode())
# 9.5.4 Discretization and quantiling
"""
Continuous values can be discretized using the cut() (bins based on values) and qcut() (bins based on sample
quantiles) functions:
"""
arr = np.random.randn(20)
# print(arr)
factor = pd.cut(arr, 4)# (1,4),(2,3)
# print(factor)
factor1 = pd.cut(arr, [-5, -1, 0, 1, 5])
# print(factor1 )
# 9.6 Function application
# 9.6.2 Row or Column-wise Function Application
"""

print(df)
print(df.apply(np.mean))
print(df.apply(np.mean, axis=1))
print(df.apply(lambda x: x.max() - x.min()))
print(df.apply(np.cumsum))
print(df.apply(np.exp))
tsdf = pd.DataFrame(np.random.randn(1000, 3), columns=['A', 'B', 'C'], index=pd.date_range('1/1/2000', periods=1000))
print(tsdf.apply(lambda x: x.idxmax()))
"""
# 9.6.3 Aggregation API
tsdf = pd.DataFrame(np.random.randn(10, 3), columns=['A', 'B', 'C'],index=pd.date_range('1/1/2000', periods=10))
tsdf.iloc[3:7] = np.nan
# print(tsdf)
print(tsdf.apply(np.sum))
# 9.10 Vectorized string methods
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
print(s.str.lower())
# 9.11 Sorting
unsorted_df = df.reindex(index=['a', 'd', 'c', 'b'], columns=['three', 'two', 'one']) # reindex 修改index
"""
print(df)
print(unsorted_df)
print(unsorted_df.sort_index())
print(unsorted_df.sort_index(ascending=False))
print(unsorted_df.sort_index(axis=1))
print(unsorted_df['three'].sort_index())
"""
# 9.11.2 By Values
df1 = pd.DataFrame({'one':[2,1,1,1],'two':[1,3,2,4],'three':[5,4,3,2]})
# print(df1.sort_values(by='two'))
# print(df1[['one', 'two', 'three']].sort_values(by=['one','two']))
s[2] = np.nan
