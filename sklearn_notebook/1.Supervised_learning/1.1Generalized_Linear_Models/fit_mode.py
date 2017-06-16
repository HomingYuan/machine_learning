# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 11:47:48 2017

@author: user
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import *

class Fit_mode(object):
    # init the data
    def __init__(self, data_x_train, data_y_train, data_x_test, data_y_test):
        self.data_x_train = data_x_train
        self.data_y_train = data_y_train
        self.data_x_test = data_x_test
        self.data_y_test = data_y_test
        
    # fit the mode 
    def get_fit_mode(self,regression_model, alpha=None):
         regr = regression_model
         regr.fit(self.data_x_train,self.data_y_train)
         return regr
         
    # print the information
    def print_parameter(self,regr):
        print('Coefficients:', regr.coef_)
        
    # plot the mode
    def plot_mode(self,regr):
            plt.scatter(self.data_x_test, self.data_y_test,  color='black')
            plt.plot(self.data_x_test, regr.predict(self.data_x_test), color='blue',linewidth=3)
            plt.xticks(())
            plt.yticks(())
            plt.show()
            


def main():
    diabetes = datasets.load_diabetes()
    # Use only one feature
    diabetes_X = diabetes.data[:, np.newaxis, 2]
    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]
    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test = diabetes.target[-20:]
    mode = Fit_mode(diabetes_X_train, diabetes_y_train, diabetes_X_test, diabetes_y_test)  # build instance
    regr = mode.get_fit_mode(LinearRegression())
    print_info = mode.print_parameter(regr)
    plot_mode = mode.plot_mode(regr)

if __name__ == '__main__':
    main()
    







           
            
            