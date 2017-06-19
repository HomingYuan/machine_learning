# -*- coding: utf-8 -*-
# 1.5. Stochastic Gradient Descent
# reference website http://scikit-learn.org/stable/modules/sgd.html
"""
Stochastic Gradient Descent (SGD) is a simple yet very efficient approach to
 discriminative learning of linear classifiers under convex loss functions such as 
 (linear) Support Vector Machines and Logistic Regression. Even though SGD has been
 around in the machine learning community for a long time, it has received a 
 considerable amount of attention just recently in the context of large-scale learning.
"""

# 1.5.1. Classification
# example 1
"""
from sklearn.linear_model import SGDClassifier
X = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = SGDClassifier(loss="hinge", penalty="l2")
clf.fit(X, y)
print('coef', clf.coef_  )
print('intercept', clf.intercept_ )
print('decision function', clf.decision_function([[2., 2.]]) )
clf1 = SGDClassifier(loss="log").fit(X, y)
print(clf1.predict_proba([[1., 1.]])) 
"""
# 1.5.2. Regression
"""
loss="squared_loss": Ordinary least squares,
loss="huber": Huber loss for robust regression,
loss="epsilon_insensitive": linear Support Vector Regression.
"""
# 1.5.3. Stochastic Gradient Descent for sparse data