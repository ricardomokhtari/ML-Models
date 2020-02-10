#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 17:39:01 2019

@author: Ricardo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree = 10) # the degree specifies the order of the polynomial
X_poly = poly_reg.fit_transform(X) # X_poly is a new matrix containing the polynomial terms x^2, x^3 etc. up to the degree
poly_reg.fit(X_poly,y)
lin_reg_2 = LinearRegression() # need to incorporate the polynomial fit into a multiple linear regression model
lin_reg_2.fit(X_poly,y)

# visualising the results - polynomial
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X,y, color = 'red')
plt.plot(X_grid,lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff - Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# predicting a new result with polynomial regression
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))

