#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:20:40 2020

@author: Ricardo
"""

# Data preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# onehot
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

X = X[:, 1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Create ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

# initialise the ANN
classifier = Sequential()

# Add input layer + first hidden layer
# NOTE: change the input dim argument depending on how many independent variables
# that you have
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Add second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Add the output layer
# NOTE: if the output has more than 2 categories, the units argument needs to be
# the same as the number of categories and the activation function needs to be softmax
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compile the Neural Network
# NOTE: Only use binary_crossnetropy if the output has 2 values, if it does not then use
# categorical_crossentropy
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# fit ANN to the training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# predicting the test set results
y_pred = classifier.predict(X_test)

# need to convert probabilities to binary values
y_pred = (y_pred > 0.5)

# making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

test_acc = ((cm[0,0] + cm[1,1]) / 2000) * 100