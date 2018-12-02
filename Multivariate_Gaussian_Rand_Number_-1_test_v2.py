#%%
# import the basic packages
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split

# define a function that generates a 1D Gaussian random number
def get_gaussian_random():
    m = 0
    while m == 0:
        m = round(np.random.random() * 100)
    # it takes no parameters - it returns a Gaussian number with mean 0 and a variance of 1. Here, m refers to the mathematical variable m. The higher the value, the more random numbers are used to generate a single Gaussian
    numbers = np.random.random(int(m))
    summation = float(np.sum(numbers))
    gaussian = (summation - m/2) / math.sqrt(m/12.0)

    return gaussian

def generate_known_gaussian(dimensions):
    count = 1000

    ret = []
    for i in range(count):
        current_vector = []
        for j in range(dimensions):
            g = get_gaussian_random()
            current_vector.append(g)
        ret.append(tuple(current_vector))
    return ret


def main():
    # generate 1000 2D dimensional Gaussian random vectors
    # define the desired mean and corariance matrix
    known = generate_known_gaussian(2)
    target_mean = np.matrix([[-1], [-1]])
    target_cov = np.matrix([[3, 0], [0, 3]])
    [eigenvalues, eigenvectors] = np.linalg.eig(target_cov)

    # produce the diagonal matrix of eigenvalues and the temporary matrix
    # Q that stores the matrix multiplication of the eigenvalues and
    # eigenvectors
    l = np.matrix(np.diag(np.sqrt(eigenvalues)))
    Q = np.matrix(eigenvectors) * l

    # loop through all the known random vectors
    x1_tweaked = []
    x2_tweaked = []
    tweaked_all = []
    for i, j in known:
        original = np.matrix([[i], [j]]).copy()
        # apply the linear transformation: first lining up the covariance and then lining up the mean
        tweaked = (Q * original) + target_mean
        # x1_tweaked holds the transformed first dimension
        # x2_tweaked holds the second dimension and tweaked_all holds the entire vector
        x1_tweaked.append(float(tweaked[0]))
        x2_tweaked.append(float(tweaked[1]))
        tweaked_all.append(tweaked)
    
    # print("x1_tweaked:\n", x1_tweaked)
    # print("x1_Transpose:\n", np.c_[x1_tweaked].T)
    # print("x2_tweaked:\n", x2_tweaked)
    # print("x2_Transpose:\n", np.c_[x2_tweaked].T)
    # print(tweaked_all)

    # plt.scatter(x1_tweaked, x2_tweaked)
    # plt.title('-1 Test Set')
    # # plt.axes((left, bottom, width, height), facecolor='w')
    # plt.axis([-8, 6, -10, 8])
    # # Plot horizontal lines at each y from xmin to xmax.
    # # hlines(y, xmin, xmax,...
    # # plt.hlines(0, -8, 6)
    # # Plot vertical lines at each x from ymin to ymax.
    # # vlines(x, ymin, ymax,...
    # # plt.vlines(0, -10, 8)
    # plt.show()

    X = np.c_[([x1_tweaked, x2_tweaked])].T
    X = np.around(X, decimals=1)

    print("Array:\n", X)
    # features need to be adjusted
    Y = [0] * 500 + [1] * 500
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2)
    # print(X_train)
    # print(Y_train)
    # print(X_test)
    # print(Y_test)
    # figure number
    fignum = 1

    # fit the model
    # Three different types of SVM-Kernels are displayed below. The polynomial and RBF(Radial Basis Function) are especially useful when the data-points are not linearly separable.

    for kernel in ('linear', 'poly', 'rbf'):
        clf = svm.SVC(C=1000, kernel=kernel, gamma=1)
        clf.fit(X_test, Y_test)

        # plot the line, the points, and the nearest vectors to the plane
        plt.figure(fignum, figsize=(4, 3))
        plt.clf()

        plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,facecolors='none', zorder=10, edgecolors='k')
        plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired, edgecolors='k')

        plt.axis('tight')
        x_min = -8
        x_max = 8
        y_min = -8
        y_max = 8

        XX, YY = np.mgrid[x_min:x_max:400j, y_min:y_max:400j]
        Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(XX.shape)
        plt.figure(fignum, figsize=(4, 3))
        plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
        plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-1, 0, 1])
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        fignum = fignum + 1
        print('The Number of Support Vectors:', clf.n_support_, 'Accuracy:', clf.score(X_test, Y_test))
    # plt.savefig('C_1.png')
    plt.show()

if __name__ == "__main__":
    main()
