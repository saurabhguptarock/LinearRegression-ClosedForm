import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

X, Y = make_regression(n_samples=400, n_features=1, n_informative=1, noise=1.8, random_state=11)

Y = Y.reshape((-1, 1))
X = (X - X.mean()) / X.std()

ones = np.ones((X.shape[0], 1))
X_ = np.hstack((X, ones))


def predict(X, theta):
    return np.dot(X, theta)


def getTheta(X, Y):
    Y = np.mat(Y)
    return np.linalg.pinv(np.dot(X.T, X)) * np.dot(X.T, Y)


theta = getTheta(X_, Y)

plt.scatter(X, Y)
plt.plot(X, predict(X_, theta), color='orange')
plt.show()
