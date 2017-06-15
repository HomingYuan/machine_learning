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
    regr = regression_model()
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
line_regression = fit_mode(LinearRegression)   
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
ridgecv= fit_mode(RidgeCV,alpha = alpha)
'''
# 1.1.3. Lasso
# The Lasso is a linear model that estimates sparse coefficients
lasso =fit_mode(Lasso,alpha=0.1)
lasso




























