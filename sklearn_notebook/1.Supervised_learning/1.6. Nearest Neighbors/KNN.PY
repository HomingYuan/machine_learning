# -*- coding: utf-8 -*-
# 1.6. Nearest Neighbors
# reference http://scikit-learn.org/stable/modules/neighbors.html
"""
sklearn.neighbors provides functionality for unsupervised and supervised 
neighbors-based learning methods. Unsupervised nearest neighbors is the 
foundation of many other learning methods, notably manifold learning and
 spectral clustering. Supervised neighbors-based learning comes in two flavors:
     classification for data with discrete labels, and regression for data with
     continuous labels.
"""
# 1.6.1. Unsupervised Nearest Neighbors
# example
"""
from sklearn.neighbors import NearestNeighbors
import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)
print('indices', indices)
print('distances', distances)
print(nbrs.kneighbors_graph(X).toarray())
"""

# 1.6.1.2. KDTree and BallTree Classes
"""
import numpy as np
from sklearn.neighbors import KDTree
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
kdt = KDTree(X, leaf_size=30, metric='euclidean')
print(kdt.query(X, k=2, return_distance=False) )
"""
# 1.6.2. Nearest Neighbors Classification
# 1.6.3. Nearest Neighbors Regression
# 1.6.4. Nearest Neighbor Algorithms
# 1.6.4.1. Brute Force
# 1.6.4.2. K-D Tree
# 1.6.4.3. Ball Tree
# 1.6.4.4. Choice of Nearest Neighbors Algorithm
# 1.6.4.5. Effect of leaf_size
# 1.6.5. Nearest Centroid Classifier

# ecample
"""
from sklearn.neighbors.nearest_centroid import NearestCentroid
import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])
clf = NearestCentroid()
clf.fit(X, y)
print(clf.predict([[-0.8, -1]]))
"""
# 1.6.5.1. Nearest Shrunken Centroid

# 1.6.6. Approximate Nearest Neighbors
# 1.6.6.1. Locality Sensitive Hashing Forest


