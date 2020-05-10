# -*- coding: utf-8 -*-
"""
Created on Sun May 10 16:57:12 2020

@author: dimitar
"""
import numpy as np
import matplotlib.pyplot as plt

# an array of ones of same dimension as x
# ones = np.ones_like(x) 

# Add a column of ones to x. hstack means stacking horizontally i.e. columnwise
# X = np.hstack((ones,x)) 

def plotData(x, y):
    
    fig, ax = plt.subplots() 
    ax.plot(x,y,'rx',markersize=10)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")

    return fig

def normalEquation(X,y):
    
    return np.dot((np.linalg.inv(np.dot(X.T,X))),np.dot(X.T,y))

def featureNormalize(X):
    return np.divide((X - np.mean(X,axis=0)),np.std(X,axis=0))

def computeCost(X, y, theta):
    m = len(y)
    J = (np.sum((np.dot(X,theta) - y)**2))/(2*m)
    return J

def gradientDescentOneVariable(X, y, theta, alpha, num_iters):
    
    m = len(y) # number of training examples
    J_history = np.zeros(num_iters)

    for i in range(num_iters):
        theta = theta - (alpha/m)*np.sum((np.dot(X,theta)-y)[:,None]*X,axis=0)
        J_history[i] = computeCost(X, y, theta)
        print('Cost function: ',J_history[i])
    
    return (theta, J_history)

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    
    m = len(y) # number of training examples
    J_history = np.zeros(num_iters)

    for i in range(num_iters):
        theta = theta - (alpha/m)*np.sum((np.dot(X,theta)-y)[:,None]*X,axis=0)
        J_history[i] = computeCost(X, y, theta)
        print('Cost function: ', J_history[i])
    
    return (theta,J_history)

