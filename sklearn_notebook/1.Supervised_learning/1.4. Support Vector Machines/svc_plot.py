# -*- coding: utf-8 -*-
# website:http://scikit-learn.org/stable/modules/svm.html
# example 1
"""
from sklearn import svm
X =[[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC()  # SUPPORT VECTORE CLASSIFICATION
clf.fit(X, y)
print('predict',clf.predict([[5, 5],[5,5]]))  # weird, all predict results are zero
print('score',clf.score([[0,0.2],[1,8]], [0,1]))
"""
"""
SVMs decision function depends on some subset of the training data,
 called the support vectors. Some properties of these support vectors 
 can be found in members support_vectors_, support_ and n_support:
"""
"""
print('support vector',clf.support_vectors_)
print('support',clf.support_ )
print('n-support',clf.n_support_)
"""
# example2
# 1.4.1.1. Multi-class classification
"""
from sklearn import svm
X = [[0], [1], [2], [3]]
Y = [0, 1, 2, 3]
clf = svm.SVC(decision_function_shape='ovo') # one:one
clf.fit(X, Y) 
dec = clf.decision_function([[1]])
print(dec.shape[1])
clf.decision_function_shape = "ovr" # one: rest
dec1 = clf.decision_function([[1]])
print(dec1.shape[1])
# On the other hand, LinearSVC implements “one-vs-the-rest” multi-class strategy, thus training n_class models. If there are only two classes, only one model is trained:
lin_clf = svm.LinearSVC()
lin_clf.fit(X, Y)
dec2 = lin_clf.decision_function([[1]])
print(dec2.shape[1])
"""
# 1.4.1.2. Scores and probabilities
# 1.4.1.3. Unbalanced problems
"""
In problems where it is desired to give more importance to certain classes or 
certain individual samples keywords class_weight and sample_weight can be used.
"""

# 1.4.2. Regression
# EXAMPLE
from sklearn import svm 
X = [[0, 0], [2, 2]]
y = [0.5, 2.5]
clf = svm.SVR()  # SUPPORT VECTOR REGRESSION
clf.fit(X, y)
print(clf.predict([[1, 1]]))

# 1.4.3. Density estimation, novelty detection
# 1.4.4. Complexity
# 1.4.5. Tips on Practical Use
"""
Avoiding data copy
Kernel cache size
Setting C
data


"""
# 1.4.6. Kernel functions
linear_svc = svm.SVC(kernel='linear')
print(linear_svc.kernel)
# 1.4.6.1. Custom Kernels
import numpy as np
from sklearn import svm
def my_kernel(X, Y):
    return np.dot(X, Y.T)
clf = svm.SVC(kernel=my_kernel)
# 1.4.6.1.2. Using the Gram matrix
# 1.4.6.1.3. Parameters of the RBF Kernel