#%%
# import the basic packages
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import pyplot as plt

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
    
    plt.scatter(x1_tweaked, x2_tweaked)
    plt.title('-1 training')
    # plt.axes((left, bottom, width, height), facecolor='w')
    plt.axis([-8, 6, -10, 8])
    # Plot horizontal lines at each y from xmin to xmax.
    # hlines(y, xmin, xmax,...
    # plt.hlines(0, -8, 6)
    # Plot vertical lines at each x from ymin to ymax.
    # vlines(x, ymin, ymax,...
    # plt.vlines(0, -10, 8)
    plt.savefig('-1_training_set')
    plt.show()
    

if __name__ == "__main__":
    main()


    