# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing the data set
dataset = pd.read_csv('Data.csv')

# create matrix of features (the independent variables)
X = dataset.iloc[:, :-1].values     # ':' means take all the lines - so we are taking all the rows and all the columns - 1

# creating the dependent variable vector
Y = dataset.iloc[:, 3].values       # taking the last column of the dataset as the independent

# Taking care of missing data by taking the mean of the values
from sklearn.preprocessing import Imputer                                # Imputer is a class
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)   # creating an object called imputer (inspect imputer to understand the parameters)
imputer = imputer.fit(X[:,1:3])                                          # the upper bound is 3 because we want to take columns 1 & 2 and this is a less than operation
X[:,1:3] = imputer.transform(X[:,1:3])                                   # the transform function replaces the missing values with the mean of the column

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder            # the class LabelEncoder
labelencoder_X = LabelEncoder()                                          # the object labelencoder_X (encoding the categorical data) has method LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])                          # the country column is now encoded
onehotencoder = OneHotEncoder(categorical_features = [0])                # using OneHotEncoder to enocde the categorical data into 3 columns
X = onehotencoder.fit_transform(X).toarray()

labelencoder_Y = LabelEncoder()                                          # using LabelEncoder because the order is relevant for 'Yes' and 'No'                                  
Y = labelencoder_Y.fit_transform(Y)

# Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split                     # the class train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0) # create all the sets in one line

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
