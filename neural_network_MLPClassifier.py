import pandas as pd
url="https://raw.githubusercontent.com/jdeng13/Machine-Learning/master/diabetes.csv"
dataset = pd.read_csv(url, error_bad_lines=False)

# another version of dataset
# dataset = pd.read_csv('/Volumes/GoogleDrive/My Drive/1. UCSC/1. 2018 Fall Quarter/CS242 Machine Learning/HWs/hw4/diabetes.csv', names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"])

# check out the data
dataset.head()
# check the detail information of the data
dataset.describe().transpose()
# check the number of features and label column
dataset.shape

# set up the data and labels
X = dataset.drop('Outcome', axis=1)
y = dataset['Outcome']

# training set and test set split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# data preprocessing
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# fit only to the training data
scaler.fit(X_train)

# apply the tranformation to the data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
from sklearn.neural_network import MLPClassifier
# 3 layers with the same number of neurons, 500 max iterations
mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
# fit the training data to our model
mlp.fit(X_train, y_train)

# training accuracies on the training set 
# start here since this is not training set or test set, modify the codes for them
print('Accuracy of the training set:', mlp.score(X, y))

import time
# time.time() 
# returns the time in seconds since the epoch as a floating point number
# time.clock()
# On Unix, return the current processor time as a floating point number expressed in seconds.
# On Windows, this function returns wall-clock seconds elapsed since the first call to this function.
print(time.time(), time.clock())

# Predictions and Evaluation
predictions = mlp.predict(X_test)

# use Scikit-Learn's built in metrics such as a classification report and confusion matrix to evaluate our model
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

# extract the MLP weights and biases after training the model, use its public attributes coefs_ and intercepts_
# coefs_ is a list of weight matrices, where weight matrix at index i represents the weights between layer i and layer i+1.
# intercepts_ is a list of bias vectors, where the vector at index i represents the bias values added to layer i+1
len(mlp.coefs_)
len(mlp.coefs_[0])
len(mlp.intercepts_[0])
