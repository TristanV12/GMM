from scipy.stats import norm, rankdata
import time
import math
import numpy as np
import pytz
import pandas as pd
from numpy.linalg import inv, pinv
from GaussianRUMGMM import *
from aggregate import *
import importlib
import functools
import scipy
import os
from collections import namedtuple

def find_bnds(x, a):
    l = len(a)
    for i in range(0, l):
        if a[i] == x:
            return False
    return True


def mixGauRUM_matlab(breaking):
    #Initialize
    _matlab_support = True
    try:
        matlab = importlib.import_module("matlab")
        matlab.engine = importlib.import_module("matlab.engine")
    except ImportError:
        _matlab_support = False
    m = len(breaking)
    #bnds = ((0.0, 1.0), (0.0, 10) for i in range(1, 2 * m + 1))
    matlabEng = None
    if _matlab_support:
        matlabEng = matlab.engine.start_matlab()
        lb = matlab.double(np.zeros((13,1)).tolist())
        ub = 5 * np.ones((13, 1))
        ub[0][0] = 1
        ub = matlab.double(ub.tolist())
        #print(lb, ub)
        A = matlab.double([])
        b = matlab.double([])
        Aeq = matlab.double([])
        beq = matlab.double([])
        Aeq_uncons = matlab.double([])
        beq_uncons = matlab.double([])

        # set matlab directory to the folder containing this module and thus also "optimize.m"
        matlabEng.cd(os.path.dirname(__file__), nargout=0)

        # generate an initial guess for the optimizer
        params_t0 = np.empty(2 * m + 1)
        ini = np.random.uniform(0,5,(2,m))
        params_t0[0] = np.random.rand()
        params_t0[1:m + 1] = ini[0]
        params_t0[m+1:] = ini[1]

        # optimization
        res = None

        tolfun = 1e-6 #default -6 #optimal -13
        tolx = 1e-10 #default -10 #optimal -13
        tolcon = 1e-6 #default -6 #optimal -9

        params_t0 = matlab.double(params_t0.tolist())
        #breaking = np.reshape(breaking, (1, m ** 2))
        breaking = matlab.double(breaking.tolist())
        res, val, fl = matlabEng.optimize("mixGauRUMobj.mixGauRUMobj_mat", breaking, params_t0, A, b, Aeq, beq, lb, ub, {"Algorithm": "interior-point", "Display": "off","TolFun": tolfun, "TolX": tolx, "TolCon": tolcon}, nargout=3)

        return res

def mixGauRUMobj(x, breaking):
    #print("Breaking: ", breaking)
    m = len(breaking) # number of alternatives
    alpha = x[0]
    cp0 = x[1:m+1]
    cp1 = x[m+1:2*m+1]
    se = 0 # objective function
    for i1 in range(0, m):
        for i2 in range(i1+1, m):
            #print(x[i1], x[i2], breaking[i1][i2], breaking[i2][i1])
            f21 = alpha * prPair(cp0[i2] - cp0[i1]) + (1 - alpha) * prPair(cp1[i2]-cp1[i1])
            f12 = alpha * prPair(cp0[i1] - cp0[i2]) + (1 - alpha) * prPair(cp1[i1]-cp1[i2])
            se += (breaking[i1][i2]*f21 - breaking[i2][i1]*f12) ** 2
            #print("SE: ", se)
    return se

#probability of one alternative is preferred to another.
#x is the difference between the means of the two alternatives
def mixPrPair(alpha, x):
    return np.sum(np.multiply(alpha, norm.cdf(x, 0, scale=math.sqrt(2))))

def mixGauRUMobj_log(x, breaking):
    #breaking = args
    #print("Breaking: ", breaking)
    #x = y - y.min() + 1
    #x = np.log(x)
    m = len(breaking) # number of alternatives
    alpha = x[0]
    cp0 = x[1:m+1]
    #cp0 = cp0 - cp0.min() + 1
    cp0 = np.log(cp0)
    cp1 = x[m+1:2*m+1]
    #cp1 = cp1 - cp1.min() + 1
    cp1 = np.log(cp1)
    se = 0 # objective function
    for i1 in range(0, m):
        for i2 in range(i1+1, m):
            #print(x[i1], x[i2], breaking[i1][i2], breaking[i2][i1])
            f21 = alpha * prPair(cp0[i2] - cp0[i1]) + (1 - alpha) * prPair(cp1[i2]-cp1[i1])
            f12 = alpha * prPair(cp0[i1] - cp0[i2]) + (1 - alpha) * prPair(cp1[i1]-cp1[i2])
            se += (breaking[i1][i2]*f21 - breaking[i2][i1]*f12) ** 2
            #print("SE: ", se)
    return se

#derivative of probability
def pfpxr(alpha_r, x_r):
    return alpha_r * np.exp(-x_r ** 2/4)/(2 * math.sqrt(math.pi))

def pfpar(x_r):
    return norm.cdf(x_r)
#second order derivative of probability
def ppfpxr(alpha_r, x_r):
    return -alpha_r * x_r * np.exp(- x_r ** 2/4)/(4 * math.sqrt(math.pi))

def ppfpxrpar(x_r):
    return -np.exp(-x_r ** 2/4)/(2 * math.sqrt(math.pi))

#To calculate the gradient of objective function
#Order of variables: alpha_1 to alpha_k, theta^(1), to theta^(k)
#k(m+1) variables in total
#alphas and thetas should be passed separately
#thetas is an k*m matrix
def gradientMixPrPair(breaking, alphas, thetas, n):
    k = thetas.shape[0]
    m = thetas.shape[1]
    l = k * ( m + 1 )

    grad = np.zeros((l, 1), float)

    for r in range(0,k):
        for i in range(0, m):
            for j in range(0, m):
                if j != i:
                    x = thetas[:,i]-thetas[:,j]
                    grad[r*m+k+i] += 2*n*(breaking[i][j]-n*mixPrPair(alphas, x))*pfpxr(alphas[r], x[r])
                    #calculate the partial derivatives to alphas
                    if j > i:
                        grad[r] += 2*n*(breaking[i][j]-n*mixPrPair(alphas, x))*pfpar(x[r])
    return grad

#To calculate the Hessian matrix
def mixHessianPrPair(breaking, alphas, thetas, n):
    k = thetas.shape[0]
    m = thetas.shape[1]
    l = k * ( m + 1 )
    hessian = np.zeros((l, l), float)

    for r0 in range(0, k):
        for r1 in range(0, k):
            for i in range(0, m):
                for j in range(0, m):
                    if j != i:
                        x = thetas[:,i] - thetas[:,j]
                        if r1 == r0:
                            #p^2 G over p theta^r_i square
                            hessian[r0*m+k+i][r0*m+k+i] += 2*n*(breaking[i][j]-n*mixPrPair(alphas, x))*ppfpxr(alphas[r0], x[r0])-2*n*n*pfpxr(alphas[r0], x[r0])**2
                            if j>i:
                                #p^2 G over p alpha^r p theta^r'_i
                                hessian[r0][r1*m+k+i] += 2*n*(breaking[i][j]-n*mixPrPair(alphas, x))*ppfpxrpar(x[r0])-2*n*n*pfpxr(alphas[r0], x[r0])*pfpar(x[r0])
                                #p^2 G over p theta^r_i p theta^r_j
                                hessian[r0*m+k+i][r0*m+k+j] = 2*n*n*(pfpxr(alphas[r0], x[r0]))**2-2*n*(breaking[i][j]-n*mixPrPair(alphas, x))*ppfpxr(alphas[r0], x[r0])
                            else:
                                #p^2 G over p theta^r_i p theta^r_j
                                hessian[r0*m+k+i][r0*m+k+j] = hessian[r0*m+k+j][r0*m+k+i]
                        else:
                            if j>i:
                                #p^2 G over p alpha^r p theta^r'
                                hessian[r0][r1*m+k+i] -= 2*n*n*pfpxr(alphas[r1],x[r1])*pfpar(x[r0])
                                #p^2 G over p theta^r_i p theta^r'_j
                                hessian[r0*m+k+i][r1*m+k+j] = 2*n*n*pfpxr(alphas[r1], x[r1])*pfpxr(alphas[r0], x[r0])
                            else:
                                #p^2 G over p theta^r_i p theta^r'_j
                                hessian[r0*m+k+i][r1*m+k+j] = hessian[r0*m+k+j][r1*m+k+i]
                #p^2 G over p alpha^r p theta^r'_i
                hessian[r1*m+k+i][r0] = hessian[r0][r1*m+k+i]
                if r1 > r0:
                    #p^2 G p theta^r_i p theta^r'_i
                    hessian[r0*m+k+i][r1*m+k+i] = -2*n*n*pfpxr(alphas[r0], x[r0])*pfpxr(alphas[r1], x[r1])
                else:
                    #p^2 G p theta^r_i p theta^r'_i
                    hessian[r0*m+k+i][r1*m+k+i] = hessian[r1*m+k+i][r0*m+k+i]
    return hessian

def trueMixBreaking(alphas, thetas):
    k = thetas.shape[0]
    m = thetas.shape[1]
    A=np.zeros((m,m), float)

    for i in range(0,m):
        for j in range(0,m):
            if j != i:
                x = thetas[:, i] - thetas[:, j]
                A[i][j] = mixPrPair(alphas, x)
    return A

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
def mixGMMGauRUM(alphas, thetas, data, m, n, k, itr=1):

    t0 = time.time() #get starting time

    l = k * (m + 1)
    vecThetaHat = np.ones((1, l), float)
    hatalphas = np.array([1/k]*k, float)
    #thetas = vecThetaHat[0][k:l].reshape(k, m)
    hatthetas = np.random.random((k, m))
    weights = np.ones((1, n), float)
    weights = weights[0]
    #breaking = dataBreaking(data, m, False, weights)
    tbreaking = trueMixBreaking(alphas, thetas)
    print(tbreaking)

    for itr in range(1,itr + 1):
        Hinv = np.linalg.pinv(mixHessianPrPair(tbreaking, hatalphas, hatthetas, n))
        Grad = gradientMixPrPair(tbreaking, hatalphas, hatthetas, n)
        #print("Gradient: ", Grad)
        vecThetaHat = (vecThetaHat.transpose() - np.dot(Hinv, Grad)).transpose()
        hatalphas = vecThetaHat[0][0:k]
        hatalphas = hatalphas/np.sum(hatalphas)
        hatthetas = vecThetaHat[0][k:l].reshape(k, m)
        hatthetas = hatthetas - np.amin(hatthetas,axis=1, keepdims=True)

    t = time.time() - t0
    #print("Time used:", t)

    return hatalphas, hatthetas
