#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Homing
@software: PyCharm Community Edition
@file: line_regression1.py
@time: 2017/6/12 21:52
"""
# reference book:scikit-learn-docs.pdf
# 2.1.1 machine learning:the problem setting
# categories 
'''
supervised leaarning
    -classification
    -regression
unsupervised learning
'''
# 2.1.2 loading an example dataset
'''

from sklearn import datasets  # 导入sklearn内部数据的库
iris = datasets.load_iris()   # 导入相关数据
digits = datasets.load_digits()  # 导入相关数据
'''
# 2.1.3 learning and predicting
'''
from sklearn import svm  # svm 支持向量机
from sklearn import datasets
digits = datasets.load_digits() 
clf = svm.SVC(gamma=0.001, C=100)
clf.fit(digits.data[:-1], digits.target[:-1])
print(clf.predict(digits.data[-1:]))
'''

# 2.1.4 Model persistence
'''
from sklearn import svm
from sklearn import datasets
import pickle  # 数据持续储存模块
clf = svm.SVC()
iris = datasets.load_iris()
x, y = iris.data, iris.target
clf.fit(x, y)
s = pickle.dumps(clf)  # 保存数据
clf2 = pickle.loads(s)  # 读取数据
print(clf2.predict(x[0:1]))
print(y[0])
from sklearn.externals import joblib # same function as pickle but more efficient
joblib.dump('filename.pkl')
joblib.load('filename.pkl')
'''
# 2.1.5
'''
import numpy as np
from sklearn import random_projection

rng = np.random.RandomState(0)
x =rng.rand(10, 2000)
x = np.array(x, dtype='float32')
print(x.dtype)  # x 数据格式
transformer = random_projection.GaussianRandomProjection()  
x_new = transformer.fit_transform(x)  # cast float 32 to float64
print(x_new.dtype)
'''
# Multiclass vs. multilabel fitting
# find more information at http://scikit-learn.org/stable/modules/multiclass.html
# example 1
'''
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
iris = datasets.load_iris()
x, y = iris.data, iris.target
print(x.shape, y.shape)
print( OneVsRestClassifier(LinearSVC(random_state=0).fit(x,y).predict(x).shape))
'''
# example 2
'''
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer  # 2d
from sklearn.preprocessing import MultiLabelBinarizer  # 3d or above


x = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]]
y = [0, 0, 1, 1, 2]

classif = OneVsRestClassifier(estimator=SVC(random_state=0))
# print(classif.fit(x, y).predict(x))

y = LabelBinarizer().fit_transform(y)
# print(classif.fit(x, y).predict(x))
y = [[0, 1], [0, 2], [1, 3], [0, 2, 3], [2, 4]]
y = MultiLabelBinarizer().fit_transform(y)
print(classif.fit(x, y).predict(x))
'''
# 2.2 A tutorial on statistical-learning for scientific data processing
# 2.2.1 Statistical learning: the setting and the estimator object in scikit-learn
# dataset
# example1 shape
'''
from sklearn import datasets
iris = datasets.load_iris()
data = iris.data
print(data.shape)  # print (150, 4) means 150 obsere value and 4 features
# example reshape
digits = datasets.load_digits()
print(digits.images.shape)
import matplotlib.pyplot as plt
plt.imshow(digits.images[-1],cmap=plt.cm.gray_r)
data = digits.images.reshape((digits.images.shape[0], -1))
print(data.shape)
'''
# estumators object has no example
# 2.2.2 Supervised learning: predicting an output variable from high-dimensional observations
# Nearest neighbor and the curse of dimensionality
'''
import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
iris_x = iris.data
iris_y = iris.target
#print(np.unique(iris_y))
#print(iris_y)
# k-Nearest neighbors classifier
# KNN (k nearest neighbors) classification example
'''
'''
np.random.seed(0)
indices = np.random.permutation(len(iris_x))  #permuation split the data randomly
iris_x_train = iris_x[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_x_test = iris_x[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(iris_x_train, iris_y_train)
print(knn.predict(iris_x_test),'\n',iris_y_test)
'''
# Linear model: from regression to sparsity
'''
diabetes = datasets.load_diabetes()
diabetes_X_train = diabetes.data[:-20]
diabetes_X_test = diabetes.data[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)
print(regr.coef_)
print(np.mean((regr.predict(diabetes_X_test)-diabetes_y_test)**2))  # residure squre
print(regr.score(diabetes_X_train , diabetes_y_train))  # fit model rate
print(regr.score(diabetes_X_test , diabetes_y_test))

# shrinkage if there are litter data points, noise in the observations induces high variance

X = np.c_[ .5, 1].T
y = [.5, 1]
test = np.c_[ 0, 2].T
regr = linear_model.LinearRegression()
'''
import matplotlib.pyplot as plt
'''
plt.figure()     
np.random.seed(0)
for _ in range(6):
    this_X = .1*np.random.normal(size=(2, 1)) + X
    regr.fit(this_X, y)
    plt.plot(test, regr.predict(test))
    plt.scatter(this_X, y, s=3)
plt.savefig('lineregression_various.pdf')
# we can see the results is various huge 
# so on this situation, we can use ridgeregression  
''' 

# ridgeregression
'''
regr = linear_model.Ridge(alpha=.1)
plt.figure()  
for _ in range(6):
 this_X = .1*np.random.normal(size=(2, 1)) + X
 regr.fit(this_X, y)
 plt.plot(test, regr.predict(test))
 plt.scatter(this_X, y, s=3)

plt.savefig('ridgeregression_stable.pdf') 
# so we can see it be more stable    
'''

# Sparsity
'''
alphas = np.logspace(-4, -1, 6)
regr = linear_model.Lasso()
scores = [regr.set_params(alpha=alpha
).fit(diabetes_X_train, diabetes_y_train
).score(diabetes_X_test, diabetes_y_test)
for alpha in alphas]
best_alpha = alphas[scores.index(max(scores))]
regr.alpha = best_alpha
regr.fit(diabetes_X_train, diabetes_y_train)
print(regr.coef_)
'''
# classification
'''
from sklearn import datasets
from sklearn import linear_model
logistic = linear_model.LogisticRegression(C=1e5)
np.random.seed(0)
indices = np.random.permutation(len(iris_x))  #permuation split the data randomly
iris_x_train = iris_x[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_x_test = iris_x[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]
logistic.fit(iris_x_train, iris_y_train)
regr = linear_model.LinearRegression()
regr.fit(iris_x_train, iris_y_train)
import matplotlib.pyplot as plt
log = plt.figure()
plt.plot(iris_x_test, logistic.predict(iris_x_test),color='r')
plt.plot(iris_x_test, regr.predict(iris_x_test),color='g')
plt.show()
'''

# Support vector machines
# Linear SVMs
'''
from sklearn import svm
indices = np.random.permutation(len(iris_x))
iris_x_train = iris_x[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_x_test = iris_x[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]
#svc = svm.SVC(kernel='linear')
svc = svm.SVC(kernel='poly',degree=3)
svc.fit(iris_x_train, iris_y_train)
import matplotlib.pyplot as plt
plt.figure()
plt.plot(iris_x_test, svc.predict(iris_x_test))
plt.show()
'''
# 2.2.3 Model selection: choosing estimators and their parameters
# Score, and cross-validated scores
'''
from sklearn import datasets, svm
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
svc = svm.SVC(C=1, kernel='linear')
print(svc.fit(X_digits[:-100], y_digits[:-100]).score(X_digits[-100:], y_digits[-100:]))
import numpy as np
X_folds = np.array_split(X_digits, 3)  # split data into 3
y_folds = np.array_split(y_digits, 3)
scores = list()
for k in range(3):
     # We use 'list' to copy, in order to 'pop' later on

    X_train = list(X_folds)

    X_test = X_train.pop(k)

    X_train = np.concatenate(X_train)

    y_train = list(y_folds)

    y_test = y_train.pop(k)

    y_train = np.concatenate(y_train)

    scores.append(svc.fit(X_train, y_train).score(X_test, y_test))
print(scores)
# Cross-validation generators
from sklearn.model_selection import KFold,cross_val_score
X = ["a", "a", "b", "c", "c", "c"]
k_fold = KFold(n_splits=3)
for train_indices, test_indices in k_fold.split(X):
    print('Train: %s | test: %s' % (train_indices, test_indices))
svc_data = [svc.fit(X_digits[train], y_digits[train]).score(X_digits[test], y_digits[test])for train, test in k_fold.split(X_digits)]
print(svc_data)
print(cross_val_score(svc, X_digits, y_digits, cv=k_fold, n_jobs=-1))
print(cross_val_score(svc, X_digits, y_digits, cv=k_fold, scoring='precision_macro'))

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import datasets, svm
digits = datasets.load_digits()
X = digits.data
y = digits.target
svc = svm.SVC(kernel='linear')
C_s = np.logspace(-10, 0, 10)
'''
# Grid-search and cross-validated estimators
# Grid-search
# 2.2.4 Unsupervised learning: seeking representations of the data
# Clustering: grouping observations together


# Principal component analysis: PCA

 # Create a signal with only 2 useful dimensions
'''
x1 = np.random.normal(size=100)
x2 = np.random.normal(size=100)
x3 = x1 + x2
X = np.c_[x1, x2, x3]
from sklearn import decomposition
pca = decomposition.PCA()
pca.fit(X)
print(pca.explained_variance_)
# As we can see, only the 2 first components are useful
pca.n_components = 2
X_reduced = pca.fit_transform(X)
print(X_reduced.shape)
'''
# Independent Component Analysis: ICA
# Generate sample data
'''
import numpy as np
from sklearn import decomposition
time = np.linspace(0, 10, 2000)
s1 = np.sin(2 * time) # Signal 1 : sinusoidal signal
s2 = np.sign(np.sin(3 * time)) # Signal 2 : square signal
S = np.c_[s1, s2]
S += 0.2 * np.random.normal(size=S.shape) # Add noise
S /= S.std(axis=0) # Standardize data
A = np.array([[1, 1], [0.5, 2]]) # Mixing matrix
X = np.dot(S, A.T) # Generate observations
# Compute ICA
ica = decomposition.FastICA()
S_ = ica.fit_transform(X) # Get the estimated sources
A_ = ica.mixing_.T
print(np.allclose(X, np.dot(S_, A_) + ica.mean_))
'''
# 2.2.5 Putting it all together

'''
from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
logistic = linear_model.LogisticRegression()
pca = decomposition.PCA()
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
# Plot the PCA spectrum
pca.fit(X_digits)
plt.figure(1, figsize=(4, 3))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_')
# Prediction
n_components = [20, 40, 64]
Cs = np.logspace(-4, 4, 3)
# Parameters of pipelines can be set using ‘__’ separated parameter names:
estimator = GridSearchCV(pipe,
dict(pca__n_components=n_components,
logistic__C=Cs))
estimator.fit(X_digits, y_digits)
plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
linestyle=':', label='n_components chosen')
plt.legend(prop=dict(size=12))
plt.savefig('pipeline_pca.pdf')
plt.show()
'''
# 2.3 Working With Text Data
# 2.3.2 Loading the 20 newsgroups dataset
'''
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

print(twenty_train.target_names)
print(len(twenty_train.data))
print("\n".join(twenty_train.data[0].split("\n")[:3]))

for t in twenty_train.target[:10]:
    print(twenty_train.target_names[t])

'''
# Tokenizing text with scikit-learn
'''
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
print(X_train_counts.shape)
print(count_vect.vocabulary_.get(u'algorithm'))
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# 2.3.4 Training a classifier
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)
for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))
'''
# Machine learning for Neuro-Imaging in Python
# http://nilearn.github.io/
# https://github.com/astroML/sklearn_tutorial

# 3.1 Supervised learning
# 3.1.1 Generalized Linear Models
# Ordinary Least Squares
'''
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit ([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
print(reg.coef_)
'''
"""
However, coefficient estimates for Ordinary Least Squares rely on the independence of the model terms. When terms
are correlated and the columns of the design matrix 𝑋 have an approximate linear dependence, the design matrix
becomes close to singular and as a result, the least-squares estimate becomes highly sensitive to random errors in the
observed response, producing a large variance. This situation of multicollinearity can arise, for example, when data
are collected without an experimental design.
"""
# Ridge Regression
'''
from sklearn import linear_model
reg = linear_model.Ridge (alpha = .5)
reg.fit ([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
print(reg.coef_)
print(reg.intercept_)
'''
# Setting the regularization parameter: generalized Cross-Validation
'''
from sklearn import linear_model
reg = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
print(reg.alpha_)
'''
# lasso
'''
from sklearn import linear_model
reg = linear_model.Lasso(alpha = 0.1)
reg.fit([[0, 0], [1, 1]], [0, 1])
print(reg.predict([[1, 1]]))
'''
# Using cross-validation
# LassoCV
# Multi-task Lasso
# LARS Lasso
from sklearn import linear_model
reg = linear_model.LassoLars(alpha=.1)
reg.fit([[0, 0], [1, 1]], [0, 1])
print(reg.coef_)
