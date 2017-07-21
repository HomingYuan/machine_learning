#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re

s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
"""
print(s.str.lower())
print(s.str.upper())
print(s.str.len())
"""
idx = pd.Index([' jack', 'jill ', ' jesse ', 'frank'])
# print(idx)
"""
print(idx.str.strip())
print(idx.str.lstrip())
print(idx.str.rstrip())
"""
df = pd.DataFrame(np.random.randn(3, 2), columns=[' Column A ', ' Column B '], index=range(3))
# print(df)
# print(df.columns.str.strip())
# print(df.columns.str.lower())
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
# print(df)
# 10.1 Splitting and Replacing Strings
s2 = pd.Series(['a_b_c', 'c_d_e', np.nan, 'f_g_h'])
"""
print(s2.str.split('_'))
print(s2.str.split('_').str.get(1))
print(s2.str.split('_').str[1])
print(s2.str.split('_', expand=True))

print(s2.str.split('_', expand=True, n=1))
print(s2.str.split('_', expand=True, n=2))
print(s2.str.rsplit('_', expand=True, n=2))
print(s2.str.rsplit('_', expand=True, n=1))
"""
s3 = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca',
 '', np.nan, 'CABA', 'dog', 'cat'])
# print(s3)
# print(s3.str.replace('^.a|dog', 'XX-XX ', case=False))
dollars = pd.Series(['12', '-$10', '$10,000'])
# print(dollars.str.replace('$', ''))
pat = r'[a-z]+'
repl = lambda m: m.group(0)[::-1]
# print(pd.Series(['foo 123', 'bar baz', np.nan]).str.replace(pat, repl))
# 10.2 Indexing with .str
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
# print(s.str[0])
# print(s.str[1])
"""
10.3 Extracting Substrings
10.3.1 Extract first match in each subject (extract)
"""
# print(pd.Series(['a1', 'b2', 'c3']).str.extract('([abc])(\d)', expand=False))
# print(pd.Series(['a1', 'b2', '3']).str.extract('([ab])?(\d)', expand=False))
# 10.4 Testing for Strings that Match or Contain a Pattern
pattern = r'[0-9][a-z]'
# print(pd.Series(['1', '2', '3a', '3b', '03c']).str.contains(pattern))
# print(pd.Series(['1', '2', '3a', '3b', '03c']).str.match(pattern))
# 10.5 Creating Indicator Variables