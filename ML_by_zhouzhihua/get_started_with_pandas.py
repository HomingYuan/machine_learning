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
# print(frame)
frame2 = frame.reindex(['a', 'b', 'c', 'd'])
# print(frame2)
states = ['Texas', 'Utah', 'California']
# print(frame.reindex(columns=states))
# print(frame.reindex(index=['a', 'b', 'c', 'd'], method='ffill',
 #columns=states))
# print(frame.ix[['a', 'b', 'c', 'd'], states])

# drop index
obj = Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
new_obj = obj.drop('c')
# print(new_obj)
# print(obj.drop(['d', 'c']))
data = DataFrame(np.arange(16).reshape((4, 4)),
 index=['Ohio', 'Colorado', 'Utah', 'New York'],
 columns=['one', 'two', 'three', 'four'])
# print(data.drop(['Colorado', 'Ohio']))
# print(data.drop('two', axis=1))
# print(data.drop('two')) # it will cause error
# print(data.drop(['two', 'four'], axis=1))
# Indexing, selection, and filtering
obj = Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
"""
print(obj)
print(obj['b'])
print(obj[1])
print(obj[2:4])
print(obj[['b', 'a', 'd']])
print(obj[obj < 2])
"""
data = DataFrame(np.arange(16).reshape((4, 4)),
 index=['Ohio', 'Colorado', 'Utah', 'New York'],
 columns=['one', 'two', 'three', 'four'])
# print(data)
# print(data['two'])
# print(data.ix['Colorado', ['two', 'three']])
"""
print(data.ix[['Colorado', 'Utah'], [3, 0, 1]])
print(data.ix[2])
print(data.ix[:'Utah', 'two'])
print(data.ix[data.three > 5, :3])
"""
s1 = Series([7.3, -2.5, 3.4, 1.5], index=['a', 'c', 'd', 'e'])
s2 = Series([-2.1, 3.6, -1.5, 4, 3.1], index=['a', 'c', 'e', 'f', 'g'])
# print(s1 + s2)
df1 = DataFrame(np.arange(9.).reshape((3, 3)), columns=list('bcd'),
 index=['Ohio', 'Texas', 'Colorado'])
df2 = DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'),
 index=['Utah', 'Ohio', 'Texas', 'Oregon'])
# print(df1 + df2)
df1 = DataFrame(np.arange(12.).reshape((3, 4)), columns=list('abcd'))
df2 = DataFrame(np.arange(20.).reshape((4, 5)), columns=list('abcde'))
"""
print(df1 + df2)
print(df1.add(df2, fill_value=0))
print(df1.reindex(columns=df2.columns, fill_value=0))
"""
arr = np.arange(12.).reshape((3, 4))
# print(arr)
# print(arr - arr[0])
frame = DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'),
 index=['Utah', 'Ohio', 'Texas', 'Oregon'])
series = frame.ix[0]
# print(frame)
# print(series)
# print(frame - series)
series2 = Series(range(3), index=['b', 'e', 'f'])
# print(frame + series2)
# Function application and mapping
