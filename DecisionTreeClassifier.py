import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# use panda's read_csv method to read CSV data file
dataset = pd.read_csv("/Volumes/GoogleDrive/My Drive/1. UCSC/1. 2018 Fall Quarter/CS242 Machine Learning/HWs/hw4/diabetes.csv")
# see the number of rows and columns in the dataset
dataset.shape
# inspect the first five records of the dataset
dataset.head()
# divide data into attributes and labels
X = dataset.drop('Outcome', axis=1)
y = dataset['Outcome']
# split the data into training and test sets
# the model_selection library of Scikit-Learn contains train_test_split method, which we'll use to randomly split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
# train the decision tree algorithm on this data and make predictions
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_depth=2, random_state=0)
# fit method
# Build a decision tree classifier from the training set (X, y)
classifier.fit(X_train, y_train)
iris = load_iris()
cross_val_score(classifier, iris.data, iris.target, cv=10)

# evaluate the algorithm and make predictions
# the predict method of the DecisionTreeClassifier class is used
y_pred = classifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Evaluate the algorithm using Mean Absolute Error, Mean Squared Error, Root Mean Squared Error
from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# apply sklearn.ensemble.RandomForrestClassifier to the training data
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)
classifier = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
classifier.fit(X, y)
print(classifier.feature_importances_)
print(classifier.predict([[0, 0, 0, 0]]))