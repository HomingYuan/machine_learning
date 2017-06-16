# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 08:43:38 2017

@author: user
"""

# 1.1  Generalized Linear Models
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import *
# Load the diabetes dataset
diabetes = datasets.load_diabetes()

# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]
# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

def fit_mode(regression_model,data_x_train=diabetes_X_train,
             data_y_train=diabetes_y_train,
             data_x_test=diabetes_X_test,
             data_y_test=diabetes_y_test,
              alpha=None):
    regr = regression_model
    regr.fit(data_x_train,data_y_train)
    print('Coefficients:', regr.coef_)
    print('Fit score:',regr.score(data_x_test,data_y_test))
    plt.scatter(data_x_test, data_y_test,  color='black')
    plt.title(str(regression_model))
    plt.plot(data_x_test, regr.predict(data_x_test), color='blue',
         linewidth=3)

    plt.xticks(())
    plt.yticks(())
    plt.show()

  # 1.1.1. Ordinary Least Squares
'''
line_regression = fit_mode(LinearRegression())   
line_regression
'''

# 1.1.2. Ridge Regression
'''
# linearregression is not stable, the output value vairous much with litter change of input
# ridge add a weight item to punish the worse data
ridge = fit_mode(Ridge,alpha=0.5)
ridge
'''
# 1.1.2.2. Setting the regularization parameter: generalized Cross-Validation
'''
alpha = [0.1, 1.0, 10.0]
ridgecv= fit_mode(RidgeCV(),alpha = alpha)
'''
# 1.1.3. Lasso
# The Lasso is a linear model that estimates sparse coefficients
'''
lasso =fit_mode(Lasso(),alpha=0.1)
lasso
'''
# 1.1.3.1. Setting regularization parameter

# The alpha parameter controls the degree of sparsity of the coefficients estimated
# 1.1.3.1.1. Using cross-validation
# 1.1.3.1.2. Information-criteria based model selection
# 1.1.4. Multi-task Lasso
# The MultiTaskLasso is a linear model that estimates sparse coefficients for multiple regression problems jointly:
# 1.1.5. Elastic Net
# ElasticNet is a linear regression model trained with L1 and L2 prior as regularizer
# 1.1.6. Multi-task Elastic Net
# The MultiTaskElasticNet is an elastic-net model that estimates sparse coefficients for multiple regression problems jointly
# 1.1.7. Least Angle Regression
# Least-angle regression (LARS) is a regression algorithm for high-dimensional data, developed by Bradley Efron, Trevor Hastie, Iain Johnstone and Robert Tibshirani.
# 1.1.8. LARS Lasso
# LassoLars is a lasso model implemented using the LARS algorithm, and unlike the implementation based on coordinate_descent, this yields the exact solution, which is piecewise linear as a function of the norm of its coefficients.
'''
lassolars = fit_mode(LassoLars(),alpha=0.1)
lassolars
'''
# 1.1.9. Orthogonal Matching Pursuit (OMP)
# OrthogonalMatchingPursuit and orthogonal_mp implements the OMP algorithm for approximating the fit of a linear model with constraints imposed on the number of non-zero coefficients (ie. the L 0 pseudo-norm).
'''
omp = fit_mode(OrthogonalMatchingPursuit())
omp

'''

# 1.1.10. Bayesian Regression
# Bayesian regression techniques can be used to include regularization parameters in the estimation procedure: the regularization parameter is not set in a hard sense but tuned to the data at hand.

# 1.1.10.1. Bayesian Ridge Regression
# BayesianRidge estimates a probabilistic model of the regression problem as described above. The prior for the parameter w is given by a spherical Gaussian
'''
BayesianRidge = fit_mode(BayesianRidge())
BayesianRidge
'''
# 1.1.10.2. Automatic Relevance Determination - ARD
# ARDRegression is very similar to Bayesian Ridge Regression, but can lead to sparser weights w [1] [2]. ARDRegression poses a different prior over w, by dropping the assumption of the Gaussian being spherical.
'''
ARDRegression = fit_mode(ARDRegression())
ARDRegression
'''
# 1.1.11. Logistic regression
'''
Logistic regression, despite its name, is a linear model for classification rather than regression. Logistic regression is also known in the literature as logit regression, maximum-entropy classification (MaxEnt) or the log-linear classifier. In this model, the probabilities describing the possible outcomes of a single trial are modeled using a logistic function.
'''
# 1.1.12. Stochastic Gradient Descent - SGD
# Stochastic gradient descent is a simple yet very efficient approach to fit linear models. It is particularly useful when the number of samples (and the number of features) is very large. The partial_fit method allows only/out-of-core learning












