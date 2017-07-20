#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

# 8.1 Series
s = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
# print(s)
# print(s.index)
# print(pd.Series(np.random.randn(5)))
d = {'a' : 0., 'b' : 1., 'c' : 2.}
# print(pd.Series(d))
# print(pd.Series(d, index=['b', 'c', 'd', 'a']))
# print(pd.Series(5., index=['a', 'b', 'c', 'd', 'e']))
# print(s[0])
# 8.1.1 Series is ndarray-like
"""
print(s[:3])
print(s[s > s.median()])
print(s[[4, 3, 1]])
"""
# 8.1.2 Series is dict-like
# print(s['a'])
# 8.1.3 Vectorized operations and label alignment with Series
"""
print(s + s)
print(s * 2)
print(np.exp(s))
print(s[:-1])
print(s[1:])
print(s[1:] + s[:-1])
"""
# 8.1.4 Name attribute
s = pd.Series(np.random.randn(5), name='something')
# print(s)
# print(s.name)
s2 = s.rename("different")
# print(s2.name)
# print((s2))
# 8.2 DataFrame
# 8.2.1 From dict of Series or dicts
d = {'one' : pd.Series([1., 2., 3.], index=['a', 'b', 'c']),
'two' : pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}
df = pd.DataFrame(d)
# print(df)
# print(pd.DataFrame(d, index=['d', 'b', 'a'], columns=['two', 'three']))
# print(df.index)
# print(df.columns)
# 8.2.2 From dict of ndarrays / lists
d = {'one' : [1., 2., 3., 4.], 'two' : [4., 3., 2., 1.]}
"""
print(pd.DataFrame(d)) # 默认range(len(d))
print(pd.DataFrame(d, index=['a', 'b', 'c', 'd']))
"""
# 8.2.3 From structured or record array
data = np.zeros((2,), dtype=[('A', 'i4'),('B', 'f4'),('C', 'a10')])
data[:] = [(1,2.,'Hello'), (2,3.,"World")]
"""
print(pd.DataFrame(data))
print(pd.DataFrame(data, index=['first', 'second']))
print(pd.DataFrame(data, columns=['C', 'A', 'B']))
"""
# 8.2.4 From a list of dicts
data2 = [{'a': 1, 'b': 2}, {'a': 5, 'b': 10, 'c': 20}]
"""
print(pd.DataFrame(data2))
print(pd.DataFrame(data2, index=['first', 'second']))
print(pd.DataFrame(data2, columns=['a', 'b']))
"""
# 8.2.5 From a dict of tuples
# 8.2.8 Column selection, addition, deletion
# print(df['one'])
df['three'] = df['one'] * df['two']
df['flag'] = df['one'] > 2
# print(df)
del df['two']
three = df.pop('three')
# print(df)
df['foo'] = 'bar'
# print(df)
df['one_trunc'] = df['one'][:2]
# print(df)
df.insert(2, 'bar', df['one']) # location,column_label,values
# print(df)
# 8.2.9 Assigning New Columns in Method Chains
# iris = pd.read_csv('data/iris.data')
# print(iris.head())
# 8.2.10 Indexing / Selection
# print(df.loc['b']) # column label
# print(df.iloc[2]) # index label
# 8.2.11 Data alignment and arithmetic
df = pd.DataFrame(np.random.randn(10, 4), columns=['A', 'B', 'C', 'D'])
df2 = pd.DataFrame(np.random.randn(7, 3), columns=['A', 'B', 'C'])
# print(df+df2)
# print(df - df.iloc[0])
index = pd.date_range('1/1/2000', periods=8)
df = pd.DataFrame(np.random.randn(8, 3), index=index, columns=list('ABC'))
# print(df)
# 8.2.12 Transposing
# print(df[:5].T)
# 8.2.13 DataFrame interoperability with NumPy functions
# print(np.exp(df))
# print(np.asarray(df))
# print(df.T.dot(df))
s1 = pd.Series(np.arange(5,10))
"""
print(s1)
print(s1.dot(s1))
"""
# 8.3 Panel
# 8.3.1 From 3D ndarray with optional axis labels
wp = pd.Panel(np.random.randn(2, 5, 4), items=['Item1', 'Item2'],
 major_axis=pd.date_range('1/1/2000', periods=5),
 minor_axis=['A', 'B', 'C', 'D'])
# print(wp)
# 8.3.2 From dict of DataFrame objects
data = {'Item1' : pd.DataFrame(np.random.randn(4, 3)),
 'Item2' : pd.DataFrame(np.random.randn(4, 2))}
# print(pd.Panel(data))
# print(pd.Panel.from_dict(data, orient='minor'))
df = pd.DataFrame({'a': ['foo', 'bar', 'baz'], 'b': np.random.randn(3)})
data = {'item1': df, 'item2': df}
panel = pd.Panel.from_dict(data, orient='minor')
# print(panel['a'])
# print(panel['b'])
# print(panel['item1']) # 会报错
# 8.3.3 From DataFrame using to_panel method
midx = pd.MultiIndex(levels=[['one', 'two'], ['x','y']], labels=[[1,1,0,0],[1,0,1,0]])
df = pd.DataFrame({'A' : [1, 2, 3, 4], 'B': [5, 6, 7, 8]}, index=midx)
# print(df.to_panel())
# 8.3.4 Item selection / addition / deletion
# print(wp['Item1'])
# 8.3.5 Transposing
# print(wp.transpose(2, 0, 1))
# 8.3.6 Indexing / Selection
"""
print(wp['Item1'])
print(wp.major_xs(wp.major_axis[2]))
print(wp.minor_axis)
print(wp.minor_xs('C'))
"""
# 8.3.7 Squeezing
"""
print(wp.reindex(items=['Item1']).squeeze()) # same as wp['Item1']
print(wp['Item1'])
"""
# 8.3.8 Conversion to DataFrame
panel = pd.Panel(np.random.randn(3, 5, 4), items=['one', 'two', 'three'],
 major_axis=pd.date_range('1/1/2000', periods=5),
minor_axis=['a', 'b', 'c', 'd'])
# print(panel.to_frame())

