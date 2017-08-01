#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

arr1 = np.array([1, 2, 3], dtype=np.float64)
arr2 = np.array([1, 2, 3], dtype=np.int32)
# print(arr1.dtype)
numeric_strings = np.array(['1.25', '-9.6', '42'], dtype=np.string_)
# print(numeric_strings.astype(float))
int_array = np.arange(10)
calibers = np.array([.22, .270, .357, .380, .44, .50], dtype=np.float64)
# print(int_array.astype(calibers.dtype))
"""
Calling astype always creates a new array (a copy of the data), even if
the new dtype is the same as the old dtype.
"""
arr = np.array([[1., 2., 3.], [4., 5., 6.]])
# print(arr * arr)
arr = np.arange(10)
# print(arr[5])
# print(arr[5:8])
arr[5:8] = 12
# print(arr)
# print(arr[5:8])
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
old_values = arr3d[0].copy()
arr3d[0] = 42
# print(arr3d)
arr3d[0] = old_values
# print(arr3d)
# print(arr3d[1,0,0])
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
# print(names)
# print(data)
# print(names == 'Bob')
# print(data[names == 'Bob'])
arr = np.empty((8, 4))
for i in range(8):
    arr[i] = i
# print(arr)
# print(arr[[4, 3, 0, 6]])
arr = np.arange(32).reshape((8, 4))
# print(arr[[1, 5, 7, 2], [0, 3, 1, 2]])
# print(arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]])
# print(arr[np.ix_([1, 5, 7, 2], [0, 3, 1, 2])])
arr = np.arange(15).reshape((3, 5))
# print(arr.T.ndim)
arr = np.random.randn(6, 3)
# print(np.dot(arr.T, arr))
arr = np.arange(16).reshape((2, 2, 4))
# print(arr.transpose((1, 0, 2)))
# print(arr.T)
# print(arr.swapaxes(1, 2))
arr = np.arange(10)
# print(np.sqrt(arr))
# print(np.exp(arr))
x = np.random.randn(8)
y = np.random.randn(8)
# print(np.maximum(x, y))
points = np.arange(-5, 5, 0.01)
