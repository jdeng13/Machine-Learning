import numpy as np
import pandas as pd

data = pd.read_csv('/Volumes/GoogleDrive/My Drive/1. UCSC/1. 2018 Fall Quarter/CS242 Machine Learning/HWs/hw4/diabetes.csv')
print(data.head())
print(data.describe())

from sklearn.model_selection import StratifiedShuffleSplit
X = data.drop('Outcome', axis=1)
y = data.Outcome
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
# Method:
# get_n_splits([X, y, groups])	
# Returns the number of splitting iterations in the cross-validator
sss.get_n_splits(X, y)

print(sss)

# Method:
# split(X, y[, groups])	
# Generate indices to split data into training and test set.
for train_index, test_index in sss.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]