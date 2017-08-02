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
print(obj4)
obj.index = ['Bob', 'Steve', 'Jeff', 'Ryan']
print(obj)