from scipy.stats import norm, rankdata
import time
import math
import numpy as np
import pytz
import pandas as pd
from numpy.linalg import inv, pinv

#probability of one alternative is preferred to another.
#x is the difference between the means of the two alternatives
def prPair(x):
    return norm.cdf(x, 0, scale=math.sqrt(2))

def GauRUMobj(x, breaking):
    #breaking = args
    #print("x:", x)
    #print("Breaking: ", breaking)
    m = len(breaking) # number of alternatives
    #print("m: ", m)
    se = 0 # objective function
    for i1 in range(0, m):
        for i2 in range(i1+1, m):
            #print(x[i1], x[i2], breaking[i1][i2], breaking[i2][i1])
            se += (breaking[i1][i2] * prPair(x[i2]-x[i1]) - breaking[i2][i1] * prPair(x[i1] - x[i2])) ** 2
            #print("SE: ", se)
    return se

#derivative of probability
def dPrPair(x):
    return np.exp(-x ** 2/4)/(2 * math.sqrt(math.pi))

#second order derivative of probability
def ddPrPair(x):
    return -x * np.exp(- x ** 2/4)/(4 * math.sqrt(math.pi))

#calculate the breakings
def dataBreaking(data, m, k, weights):
    breaking = np.zeros((k, m, m), float)
    n = len(data)
    for r in range(0, k):
        for j in range(0, n):
            for i1 in range(0, m):
                for i2 in range(i1+1, m):
                    breaking[r][data[j, i1]][data[j, i2]] += weights[r, j]
    return breaking

def freqBreaking(weights, m, k):
    breaking = np.zeros((k, m, m), float)
    for r in range(0, k):
        for vote, freq in weights[r].items():
            for i1 in range(0, m):
                for i2 in range(i1+1, m):
                    breaking[r][int(vote[i1]), int(vote[i2])] += freq
    return breaking

#for debugging purpose
def trueBreaking(Mu):
    m=len(Mu)
    A=np.zeros((m,m), float)

    for i in range(0,m):
        for j in range(0,m):
            x = Mu[i] - Mu[j]
            A[i,j] = norm.cdf(x, 0, scale=math.sqrt(2))
    np.fill_diagonal(A,0)
    return A

#To calculate the gradient of objective function
def gradientPrPair(theta, breaking):
    #theta = theta[0]
    m = len(breaking)
    n = breaking[0][1] + breaking[1][0]
    grad = np.zeros((1, m), float)
    grad = grad[0]
    for i in range(0, m):
        for j in range(0, m):
            if j != i:
                x = theta[i] - theta[j]
                grad[i] += 2 * n * (breaking[i][j] - n * prPair(x)) * dPrPair(x)
    return grad

#To calculate the Hessian matrix
def hessianPrPair(theta, breaking):
    #theta = theta[0]
    m = len(breaking)
    n = breaking[0][1] + breaking[1][0]
    hessian = np.zeros((m, m), float)
    for i in range(0, m):
        for j in range(0, m):
            if j != i:
                x = theta[i] - theta[j]
                hessian[i][i] += 2*n*(breaking[i][j]-n*prPair(x))*ddPrPair(x) - 2*n*n*(dPrPair(x))**2
                if j > i:
                    hessian[i][j] = 2*n*n*dPrPair(x)**2-2*n*ddPrPair(breaking[i][j]-n*prPair(x))
                else:
                    hessian[i][j] = hessian[j][i]
    return hessian

#' GMM Method for Estimating Random Utility Model wih Normal dsitributions
#'
#' @param Data.pairs data broken up into pairs
#' @param m number of alternatives
#' @param itr number of itrations to run
#' @param Var indicator for difference variance (default is FALSE)
#' @param prior magnitude of fake observations input into the model
#' @return Estimated mean parameters for distribution of underlying normal (variance is fixed at 1)
#' @export
#' @examples
#' data(Data.Test)
#' Data.Test.pairs <- Breaking(Data.Test, "full")
#' Estimation.Normal.GMM(Data.Test.pairs, 5)
def GMMGaussianRUM(init, breaking, m, itr=1000):

    t0 = time.time() #get starting time

    muhat = np.ones((1, m), float)
    #muhat[0] = init
    Grad = np.empty((1, m), float)
    print("GMM Itr:  ", end='')

    for it in range(1,itr + 1):
        print("\b"*len(str(it-1)) + str(it), end='')
        noise = 0
        try:
            Hinv = np.linalg.inv(hessianPrPair(muhat[0], breaking))
        except np.linalg.linalg.LinAlgError:
            noise = np.random.uniform(-0.05, 0.05, (m,))
            print("Singular Hessian! Fixed steps will be used.")
            Hinv = 0.00 * np.identity(m)
        #print(Hinv)
        Grad[0] = gradientPrPair(muhat[0], breaking)
        muhatp = (muhat.transpose() - np.dot(Hinv, Grad.transpose())).transpose() + noise
        #muhat -= 0.01 * Grad
        diff = (np.sum(muhat) - np.sum(muhatp))/m
        muhatp += diff
        se = np.sum((muhat - muhatp) ** 2)
        if se <= 1e-6:
            break
        else:
            muhat = muhatp
        #muhat = muhat - np.log(np.sum(np.exp(muhat)))
    print()
    t = time.time() - t0
    #print("Time used:", t)

    return muhatp
