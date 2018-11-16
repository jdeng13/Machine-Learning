import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
data = pd.read_csv('/Volumes/GoogleDrive/My Drive/1. UCSC/1. 2018 Fall Quarter/CS242 Machine Learning/HWs/hw4/diabetes.csv')
print(data.head())

y = data.Age
X = data.drop('Age', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
print("\nX_train:\n")
print(X_train.head())
print(X_train.shape)

print("\nX_test:\n")
print(X_test.head())
print(X_test.shape)
