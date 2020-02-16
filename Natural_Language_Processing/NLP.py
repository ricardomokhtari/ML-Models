"""

This script is an implementation of an NLP model that predicts the sentiment
of restaurant reviews.

First half of scripts is data cleaning (processing of review text into sparse
matrix)

Second half is fitting random forest regression on sparse matrix

Accuracy: 87%

"""

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# Import the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', 
                      quoting = 3) # the quoting parameter tells pandas to ignore
                                   # the double quotes in the text

# Cleaning the text
nltk.download('stopwords')

# corpus contains all cleaned reviews
corpus = []

for i in range(len(dataset)):
    # need to remove all characters that are not letters
    # replace the characters with a space 
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    
    # convert all to lower case
    review = review.lower()
    
    review = review.split()
    
    # apply stemming
    ps = PorterStemmer()
    
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    
    review = ' '.join(review)
    
    corpus.append(review)

# creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer

# use count vectoriser to create sparse matrix of words
# use max features to reduce sparsity of matrix
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()

# dependent variable (liked vs not liked)
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Create classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# predicting the test set results
y_pred = classifier.predict(X_test)

# making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# calculate test set accuracy
acc = cm[0,0] + cm[1,1] / sum(sum(cm))
print(acc)