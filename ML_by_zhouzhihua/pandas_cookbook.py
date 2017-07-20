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
import itertools
import functools
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
# print(newseries)
aValue = 43.0
df = pd.DataFrame({'AAA' : [4,5,6,7], 'BBB' : [10,20,30,40],'CCC' : [100,50,-30,-50]})
# print(df.loc[(df.CCC-aValue).abs().argsort()])
# 7.2 Selection
# Using both row labels and value conditionals
# print(df[(df.AAA <= 6) & (df.index.isin([0,2,4]))]) # choose by value then by index

data = {'AAA' : [4,5,6,7], 'BBB' : [10,20,30,40],'CCC' : [100,50,-30,-50]}
df = pd.DataFrame(data=data,index=['foo','bar','boo','kar'])
# print(df)
# print(df.loc['bar':'kar']) # loc slice like dictionary, choose index
# print(df.iloc[0:3])  # choose row, position oriented
# print(df.loc[0:3]) # not allowed
df2 = pd.DataFrame(data=data,index=[1,2,3,4])
# print(df2)
# print(df2.iloc[1:3]) # position-oriented
# print(df2.loc[1:3]) # Label-oriented,including both start and end number

# 7.2.2 Panels
rng = pd.date_range('1/1/2013',periods=100,freq='D')
data = np.random.randn(100, 4)
cols = ['A','B','C','D']
df1, df2, df3 = pd.DataFrame(data, rng, cols), pd.DataFrame(data, rng, cols),pd.DataFrame(data, rng, cols)
pf = pd.Panel({'df1':df1,'df2':df2,'df3':df3});
# print(pf)
pf = pf.transpose(2,0,1)
# print(pf)
pf['E'] = pd.DataFrame(data, rng, cols)
pf = pf.transpose(1,2,0)
# print(pf)
pf.loc[:,:,'F'] = pd.DataFrame(data, rng, cols)
# print(pf)
df = pd.DataFrame({'AAA' : [1,1,1,2,2,2,3,3], 'BBB' : [2,1,3,4,5,1,2,3]})
"""
print(df)
print(df.loc[df.groupby("AAA")["BBB"].idxmin()])
print(df.sort_values(by="BBB").groupby("AAA", as_index=False).first())
"""
# 7.3 MultiIndexing
df = pd.DataFrame({'row' : [0,1,2],
 'One_X' : [1.1,1.1,1.1],
 'One_Y' : [1.2,1.2,1.2],
 'Two_X' : [1.11,1.11,1.11],
 'Two_Y' : [1.22,1.22,1.22]})
# print(df)
df = df.set_index('row')
# print(df)
df.columns = pd.MultiIndex.from_tuples([tuple(c.split('_')) for c in df.columns])
# print(df)
df = df.stack(0).reset_index(1)
# print(df)
cols = pd.MultiIndex.from_tuples([ (x,y) for x in ['A','B','C'] for y in ['O','I']])
df = pd.DataFrame(np.random.randn(2,6),index=['n','m'],columns=cols)
# print(df)
df = df.div(df['C'],level=1)
# print(df)
# 7.3.2 Slicing
coords = [('AA','one'),('AA','six'),('BB','one'),('BB','two'),('BB','six')]
index = pd.MultiIndex.from_tuples(coords)
df = pd.DataFrame([11,22,33,44,55],index,['MyData'])
# print(df)
"""
print(df.xs('BB',level=0,axis=0))
print(df.xs('six',level=1,axis=0))
"""
index = list(itertools.product(['Ada','Quinn','Violet'],['Comp','Math','Sci']))
headr = list(itertools.product(['Exams','Labs'],['I','II']))
indx = pd.MultiIndex.from_tuples(index,names=['Student','Course'])
cols = pd.MultiIndex.from_tuples(headr)
data = [[70+x+y+(x*y)%3 for x in range(4)] for y in range(9)]
df = pd.DataFrame(data,indx,cols)
# print(df)
All = slice(None)
df.loc['Violet']
# print(df.loc[(All,'Math'),All])
"""
print(df.loc[(slice('Ada','Quinn'),'Math'),All])
print(df.loc[(All,'Math'),('Exams')])
print(df.loc[(All,'Math'),(All,'II')])
"""
# print(df.sort_values(by=('Labs', 'II'), ascending=False))
# 7.4 Missing Data
df = pd.DataFrame(np.random.randn(6,1), index=pd.date_range('2013-08-01',periods=6, freq='B'), columns=list('A'))
# print(df)
df.loc[df.index[3], 'A'] = np.nan
"""
print(df)
print(df.reindex(df.index[::-1]).ffill())
"""
df = pd.DataFrame({'animal': 'cat dog cat fish dog cat cat'.split(), 'size': list('SSMMMLL'),
                   'weight': [8, 10, 11, 1, 20, 12, 12],'adult' : [False] * 5 + [True] * 2})
# print(df)
# print(df.groupby('animal').apply(lambda subf: subf['size'][subf['weight'].idxmax()]))
gb = df.groupby(['animal'])
# print(gb.get_group('cat'))
df = pd.DataFrame({'A' : [1, 1, 2, 2], 'B' : [1, -1, 1, 2]})
gb = df.groupby('A')
# print(gb)
def replace(g):
    mask = g < 0
    g.loc[mask] = g[~mask].mean()
    return g
# print(gb.transform(replace))
# 7.5 Grouping
df = pd.DataFrame({'animal': 'cat dog cat fish dog cat cat'.split(),
'size': list('SSMMMLL'),
'weight': [8, 10, 11, 1, 20, 12, 12],
'adult' : [False] * 5 + [True] * 2})
"""
# print(df)
print(df.groupby('animal').apply(lambda subf: subf['size'][subf['weight'].idxmax()]))
print(df.groupby('animal').apply(lambda subf: subf['size'][subf['weight'].idxmin()]))
"""
gb = df.groupby(['animal'])
# print(gb.get_group('dog'))
# 通过groupby获得column下面的items，然后通过get_group（item）获得具体的item

def GrowUp(x):
    avg_weight = sum(x[x['size'] == 'S'].weight * 1.5)
    avg_weight += sum(x[x['size'] == 'M'].weight * 1.25)
    avg_weight += sum(x[x['size'] == 'L'].weight)
    avg_weight /= len(x)
    return pd.Series(['L',avg_weight,True], index=['size', 'weight', 'adult'])

expected_df = gb.apply(GrowUp)
# print(expected_df)
S = pd.Series([i / 100.0 for i in range(1,11)])
def CumRet(x,y):
     return x * (1 + y)
def Red(x):
    return functools.reduce(CumRet, x, 1.0)
"""
print(S)
print(S.expanding().apply(Red))
"""
df = pd.DataFrame({'A' : [1, 1, 2, 2], 'B' : [1, -1, 1, 2]})
gb = df.groupby('A')
#print(df)
def replace(g):
     mask = g < 0
     g.loc[mask] = g[~mask].mean()
     return g
# print(gb.transform(replace))
df = pd.DataFrame({'code': ['foo', 'bar', 'baz'] * 2,
 'data': [0.16, -0.21, 0.33, 0.45, -0.59, 0.62],
 'flag': [False, True] * 3})
code_groups = df.groupby('code')
agg_n_sort_order = code_groups[['data']].transform(sum).sort_values(by='data')
sorted_df = df.loc[agg_n_sort_order.index]
# print(sorted_df)
rng = pd.date_range(start="2014-10-07",periods=10,freq='2min')
ts = pd.Series(data = list(range(10)), index = rng)
def MyCust(x):
     if len(x) > 2:
         return x[1] * 1.234
     return pd.NaT

mhc = {'Mean' : np.mean, 'Max' : np.max, 'Custom' : MyCust}
# print(ts.resample("5min").apply(mhc))
# print(ts)
df = pd.DataFrame({'Color': 'Red Red Red Blue'.split(), 'Value': [100, 150, 50, 50]})
print(df)
df['Counts'] = df.groupby(['Color']).transform(len)
print(df)