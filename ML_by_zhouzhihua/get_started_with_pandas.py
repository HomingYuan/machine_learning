#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pandas import Series, DataFrame
import pandas as pd
import numpy as np

obj = Series([4, 7, -5, 3])
"""
print(obj)
print(obj.values)
print(obj.index)
"""
obj2 = Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
"""
print(obj2)
print(obj2[['c', 'a', 'd']])
print(obj2[obj2 > 0])
print(obj2 * 2)
print('b' in obj2)
"""
sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = Series(sdata)
# print(obj3)
states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = Series(sdata, index=states)
"""
print(obj4)
print(pd.isnull(obj4))
print(pd.notnull(obj4))
print(obj3 + obj4)
"""
obj4.name = 'population'
obj4.index.name = 'state'
# print(obj4)
obj.index = ['Bob', 'Steve', 'Jeff', 'Ryan']
# print(obj)

# DataFrame
# create DataFrame
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
'year': [2000, 2001, 2002, 2001, 2002],
'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
frame = DataFrame(data)
# print(frame)
# print(DataFrame(data, columns=['year', 'state', 'pop']))
frame2 = DataFrame(data, columns=['year', 'state', 'pop', 'debt'],
 index=['one', 'two', 'three', 'four', 'five'])
# print(frame2)
# print(frame2.columns)
# print(frame2['state'])
# print(frame2.year)
# print(frame2.ix['three'])
frame2['debt'] = 16.5
# print(frame2)
frame2['debt'] = np.arange(5.)
# print(frame2)
val = Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
frame2['debt'] = val
# print(frame2)
frame2['eastern'] = frame2.state == 'Ohio'
# print(frame2)
del frame2['eastern']
# print(frame2)
pop = {'Nevada': {2001: 2.4, 2002: 2.9},
 'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
frame3 = DataFrame(pop)
# print(frame3)
# print(frame3.T)
# print(DataFrame(pop, index=[2001, 2002, 2003]))
pdata = {'Ohio': frame3['Ohio'][:-1],
 'Nevada': frame3['Nevada'][:2]}
# print(DataFrame(pdata))
frame3.index.name = 'year'
frame3.columns.name = 'state'
# print(frame3)
# print(frame3.values)
obj = Series(range(3), index=['a', 'b', 'c'])
index = obj.index
# print(index)
index = pd.Index(np.arange(3))
obj2 = Series([1.5, -2.5, 0], index=index)
# print(obj2.index is index)
obj = Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
# print(obj)
obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])
# print(obj2)
# print(obj.reindex(['a', 'b', 'c', 'd', 'e'], fill_value=0))
obj3 = Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
# print(obj3.reindex(range(6), method='ffill'))
frame = DataFrame(np.arange(9).reshape((3, 3)), index=['a', 'c', 'd'],
 columns=['Ohio', 'Texas', 'California'])