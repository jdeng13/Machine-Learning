# importing Needed Packages
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
%matplotlib inline

# Downloading Data
# use !wget to download the data from IBM Object Storage
!wget -O FuelConsumption.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv

# Reading the data in
df = pd.read_csv("FuelConsumption.csv")
# take a look at the dataset
df.head()

# select some features that we want to use for regression
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
cdf.head(9)

# plot Emission values with respect to Engine size
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# Creating train and test dataset
# Train/Test Split will split the dataset into training and testing sets respectively, which are mutually exclusive
# Testing dataset is not part of the dataset that have been used to train the data, which is out-of-sample, this way is more realistic for real world problems.
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# Train data distribution
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# Multiple Regression Model: there are more than one independent varaible, thus called multiple regression, which is the extension of simple linear regression model

from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(x, y)
# The coefficients
# Coefficient and Intercept are the parameters of the fit line. sklearn can estimate them from our data. Scikit-learn uses plain Ordinary Least Squares method to solve thid problem
# Ordinary Least Squares (OLS) can find the best parameters using the following methods:
    # - Solving the model parameters analytically using closed-form equations
    # - Using an optimization algorithm (Gradient Descent, Stochastic Gradient Descent, Newton's Method, etc.)
print('Coefficients:', regr.coef_)

# Prediction
y_hat = regr.predict(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
x = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
y = np.asanyarray(test[['CO2EMISSIONS']])
print("Residual sum of squres:%.2f" % np.mean((y_hat - y) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x, y))
