#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
s = pd.Series([1,3,5,np.nan,6,8]) # pd.Series
# print(s)
dates = pd.date_range('20130101',periods=6) # time series
# print(dates)
df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=list('ABCD'))  # pd.DataFrame
# print(df)
# print(df.head())  # first 5 items
# print(df.tail(3)) # last 3 items
# print(df.index) # index
# print(df.columns) # columns items
# print(df.values) # values
# print(df.describe())
# print(df.T.describe()) # transpose
# print(df.T.values) # transpose
# print(df.sort_index(axis=1, ascending=False)) # sorting
# print(df.sort(columns='B'))
"""
print(df['A'])
print(df[0:3])
print(df['20130102':'20130104'])
"""
# print(df.loc[dates[0]])
# print(df.loc[:,['A','B']])
"""
print(df[df.A > 0])
print(df[df > 0])
print(df.loc['20130102':'20130104',['A','B']]) # 多次筛选数据

print(df.iloc[0:2]) # 对行进行筛选
print(df.iloc[3:5,0:2]) # 多次进行切片，先对行切片，然后对列切片
"""
s1 = pd.Series([1,2,3,4,5,6],index=pd.date_range('20130102',periods=6))
# print(s1)
df['F'] = s1
# print(df)
df2 = df.copy()
# print(df2)
df1 = df.reindex(index=dates[0:4],columns=list(df.columns) + ['E'])
# print(df1)
df1.loc[dates[0]:dates[1],'E'] = 1
"""
print(df1)
print(df1.dropna(how='any')) # df1本身不改变 处理缺省值
print(df1) # df1本身不改变
"""
# print(df1.fillna(value=5)) # 处理缺省值
# print(df.mean()) # 沿着y轴方向统计
# print(df.mean(1)) # 沿着x轴方向统计
s = pd.Series([1,3,5,np.nan,6,8],index=dates).shift(2)
# print(s)
# print(df.sub(s,axis='index'))
# print(df.apply(np.cumsum))
# print(df.apply(lambda x: x.max() - x.min()))
s = pd.Series(np.random.randint(0,7,size=10))
# print(s)
# print(s.value_counts()) # 对数值进行统计
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
# print(s.str.lower()) # 对字母进行处理，小写化
df = pd.DataFrame(np.random.randn(10, 4))
# print(df)
pieces = [df[:3], df[3:7], df[7:]]
# print(pieces)
# print(pd.concat(pieces))
left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})
# print(left)
# print(right)
# print(pd.merge(left, right, on='key'))
df = pd.DataFrame(np.random.randn(8, 4), columns=['A','B','C','D'])
# print(df)
s = df.iloc[3]
# print(s)
# print(df.append(s, ignore_index=True))

df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
'foo', 'bar', 'foo', 'foo'],
'B' : ['one', 'one', 'two', 'three',
'two', 'two', 'one', 'three'],
'C' : np.random.randn(8), 'D' : np.random.randn(8)})
# print(df)
# print(df.groupby('A').sum())
# print(df.groupby(['A','B']).sum())

df = pd.DataFrame({'A' : ['one', 'one', 'two', 'three'] * 3,
'B' : ['A', 'B', 'C'] * 4,
'C' : ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
'D' : np.random.randn(12),
'E' : np.random.randn(12)})
# print(df)
# print(pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C']))
rng = pd.date_range('1/1/2012', periods=100, freq='S')
# print(rng)
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
# print(ts)
# print(ts.resample('5Min').sum())
rng1 = pd.date_range('3/6/2012 00:00', periods=5, freq='D')
ts1 = pd.Series(np.random.randn(len(rng1)), rng1)
ts_utc = ts.tz_localize('UTC')
# print(ts_utc)
ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
# ts.plot()
df.to_csv('foo.csv')
t = pd.read_csv('foo.csv')
# print(t)
df.to_hdf('foo.h5','df')
t1 = pd.read_hdf('foo.h5','df')
# print(t1)
df.to_excel('foo.xlsx', sheet_name='Sheet1')
t2 = pd.read_excel('foo.xlsx', 'Sheet1', index_col=None, na_values=['NA'])
print(t2)

