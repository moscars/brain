import sys 
sys.path.append('..')
from neuralnetwork import Brain
import numpy as np
import matplotlib.pyplot as plt
import random

np.random.seed(950)
random.seed(950)

# make up a dataset
from sklearn.datasets import make_circles
X, y = make_circles(noise=0.2, factor=0.5, random_state=1)

newY = []
for _y in y:
    if _y == 1:
        newY.append([1, 0])
    else:
        newY.append([0, 1])

# visualize in 2D
plt.figure(figsize=(5,5))
plt.scatter(X[:,0], X[:,1], c=y, s=20, cmap='jet')

# learn
brain = Brain([2, 5, 4, 2], "Sigmoid")
brain.learn(1000, 0.05, X, newY, 20)

# visualize decision boundary
h = 0.05
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Xmesh = np.c_[xx.ravel(), yy.ravel()]
inputs = [xrow for xrow in Xmesh]
predictions = [brain.classify(x) for x in inputs]
Z = np.array([s[0] > s[1] for s in predictions])
Z = Z.reshape(xx.shape)

fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()