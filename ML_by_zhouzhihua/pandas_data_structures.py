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