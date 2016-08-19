from scipy.stats import norm, rankdata
import time
import math
import numpy as np
import pytz
import pandas as pd

def delta(Mu, SD=0, Var=False):
    Mu = Mu[0]
    m=len(Mu)
    A=np.zeros((m,m), float)

    if not Var:
        for i in range(0,m):
            for j in range(0,m):
                A[i,j]=Mu[i]-Mu[j]
    if Var:
        for i in range(0,m):
            for j in range(0,m):
                A[i,j]=2**.5*(Mu[i]-Mu[j])/(SD[i]**2+SD[j]**2)**.5
    # print(A)
    return A

def f(Mu, SD=0, Var = False):
    Mu = Mu[0]
    m=len(Mu)
    A=np.zeros((m,m), float)
    if not Var:
        for i in range(0,m):
            for j in range(0,m):
                A[i,j]=norm.cdf(Mu[i]-Mu[j],loc=0,scale=math.sqrt(2))
        np.fill_diagonal(A, 1-np.sum(A, axis=0))
  
    if Var:
        for i in range(0,m):
            for j in range(0,m):
                A[i,j]=norm.cdf(Mu[i]-Mu[j],0,scale=math.sqrt(SD[i]**2+SD[j]**2))
        np.fill_diagonal(A, 1-np.sum(A, axis=0))
    np.fill_diagonal(A,0)
    return A

# Converts a pairwise count matrix into a probability matrix
# 
# @param C original pairwise count matrix
# @return a pairwise probability matrix
# @export
# @examples
# C= matrix(c(2,4,3,5),2,2)
# normalizeC(C)
def normalizeC(C):
    normalized = C / (C + np.transpose(C))
    np.fill_diagonal(normalized, 0)
    return normalized

#' Generate a matrix of pairwise wins
#' 
#' This function takes in data that has been broken up into pair format.
#' The user is given a matrix C, where element C[i, j] represents 
#' (if normalized is FALSE) exactly how many times alternative i has beaten alternative j
#' (if normalized is TRUE)  the observed probability that alternative i beats j
#' 
#' @param Data.pairs the data broken up into pairs
#' @param m the total number of alternatives
#' @param weighted whether or not this generateC should use the third column of Data.pairs as the weights
#' @param prior the initial "fake data" that you want to include in C. A prior 
#' of 1 would mean that you initially "observe" that all alternatives beat all
#' other alternatives exactly once.
#' @param normalized if TRUE, then normalizes entries to probabilities
#' @return a Count matrix of how many times alternative i has beat alternative j
#' @export
#' @examples
#' data(Data.Test)
#' Data.Test.pairs <- Breaking(Data.Test, "full")
#' generateC(Data.Test.pairs, 5)
def generateC(DataPairs, m, weighted = False, prior = 0, normalized = True):
    diag = np.zeros((m, m), int)
    np.fill_diagonal(diag,1)
    # C is the transition matrix, where C[i, j] denotes the number of times that
    C = np.zeros((m, m), int) - prior * m * diag
  
    for i in DataPairs:
        C[i[0] - 1, i[1] - 1] += 1

    if normalized:
        return normalizeC(C)
    else:
        return C


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
def EstimationNormalGMM(DataPairs, m, itr=1000, Var=False, prior=0):
    
    t0 = time.time() #get starting time
  
    sdhat = np.ones((1,m), float)
    muhat = np.ones((1,m), float)
    C = generateC(DataPairs, m, prior)

    if not Var:
        for itr in range(1,itr + 1):
            alpha = 1/itr
            muhat = muhat + alpha *(np.exp((-delta(muhat)**2)/4)*(C - f(muhat))).sum(axis=1)
            muhat = muhat - muhat.min()
  
    if Var:
        for itr in range(1,itr + 1):
            alpha = 1/itr
            muhat = muhat + alpha *(exp(-delta(muhat,sdhat,Var=Var)^2/4)*(C - f(muhat,sdhat,Var=Var))).sum(axis=1)
            sdhat = abs(sdhat - alpha *rowSums(VarMatrix(sdhat)^(-2)*exp(-delta(muhat,sdhat,Var=Var)^2/4)*(C - f(muhat,sdhat,Var=TRUE))))
            muhat = muhat - muhat.min()
            sdhat[1] = 1
  
    t = time.time() - t0
  
    params = []
    for i in range(0,m):
        params.append(dict(Mean = muhat[0, i], SD = sdhat[0, i]))
    # print(dict(m = m, order = rankdata(-muhat[0,]), Mean = muhat[0,], SD = sdhat[0,], Time = t, Parameters = params))
    return dict(m = m, order = (-muhat[0,]).ravel().argsort(), Mean = muhat[0,], SD = sdhat[0,], Time = t, Parameters = params)