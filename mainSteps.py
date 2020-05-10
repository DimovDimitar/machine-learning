# -*- coding: utf-8 -*-
"""
Created on Sun May 10 17:18:45 2020

@author: dimit
"""
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from linRegFunctions import *

data = pd.read_csv("ex1data2.txt",names=["size","bedrooms","price"])
s = np.array(data.size)
r = np.array(data.bedrooms)
p = np.array(data.price)
m = len(r) 
s = np.vstack(s)
r = np.vstack(r)
X = np.hstack((s,r))

print(" size = ", s[:15],"\n"," bedrooms = ", r[:10], "\n")

X = featureNormalize(X)
X = np.hstack((np.ones_like(s),X))

alpha = 0.03
num_iters = 400
theta = np.zeros(3)

# Multiple Dimension Gradient Descent
theta, hist = gradientDescent(X, p, theta, alpha, num_iters)

# Plot convergence graph
fig = plt.figure()
ax = plt.subplot(111)
plt.plot(np.arange(len(hist)),hist ,'-b')
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()

print('Theta computed from gradient descent: \n')
print(theta,'\n')

# Estimate the price of a 1650 sq-ft, 3 br house
#the first column of X is all-ones.it doesnot need to be normalized.
normalized_specs = np.array([1,((1650-s.mean())/s.std()),((3-r.mean())/r.std())])
price = np.dot(normalized_specs,theta) 

print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n ',
      price)
input('Program paused. Press enter to continue.\n')

