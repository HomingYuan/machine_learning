#!/usr/bin/env python
# -*- coding: utf-8 -*-
# http://scikit-learn.org/stable/modules/feature_selection.html
# 1.13.1. Removing features with low variance
"""
from sklearn.feature_selection import VarianceThreshold
X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
print(sel.fit_transform(X))
"""
# 1.13.2. Univariate feature selection

"""
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
iris = load_iris()
X, y = iris.data, iris.target
print(X.shape)
X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
print(X_new)
"""
 # 1.13.3. Recursive feature elimination
 # 1.13.4. Feature selection using SelectFromModel
 
 # 1.13.4.1. L1-based feature selection
 # 1.13.4.2. Randomized sparse models
 
 # 1.13.4.3. Tree-based feature selection
 # 1.13.5. Feature selection as part of a pipeline