#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 1.12.1. Multilabel classification format
"""
from sklearn.preprocessing import MultiLabelBinarizer
y = [[2, 3, 4], [2], [0, 1, 3], [0, 1, 2, 3, 4], [0, 1, 2]]
print(MultiLabelBinarizer().fit_transform(y))
"""
# 1.12.2. One-Vs-The-Rest

# 1.12.2.1. Multiclass learning
"""
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
iris = datasets.load_iris()
X, y = iris.data, iris.target
print(OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X))
"""
# 1.12.2.2. Multilabel learning
# 1.12.3. One-Vs-One
# 1.12.3.1. Multiclass learning
"""
from sklearn import datasets
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
iris = datasets.load_iris()
X, y = iris.data, iris.target
print(OneVsOneClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X))
"""
# 1.12.4. Error-Correcting Output-Codes

# 1.12.4.1. Multiclass learning
"""
from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
X, y = make_regression(n_samples=10, n_targets=3, random_state=1)
print(MultiOutputRegressor(GradientBoostingRegressor(random_state=0)).fit(X, y).predict(X))
"""

# 1.12.6. Multioutput classification

from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import numpy as np
X, y1 = make_classification(n_samples=10, n_features=100, n_informative=30, n_classes=3, random_state=1)
y2 = shuffle(y1, random_state=1)
y3 = shuffle(y1, random_state=2)
Y = np.vstack((y1, y2, y3)).T
n_samples, n_features = X.shape # 10,100
n_outputs = Y.shape[1] # 3
n_classes = 3
forest = RandomForestClassifier(n_estimators=100, random_state=1)
multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
print(multi_target_forest.fit(X, Y).predict(X))