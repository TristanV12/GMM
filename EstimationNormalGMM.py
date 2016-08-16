from scipy.stats import norm

def delta(Mu, SD=0, Var=False):
    m=length(Mu)
    A=zeros((m,m), int)

    if not Var:
        for i in range(1,m + 1):
            for j in range(1,m + 1):
                A[i,j]=Mu[i]-Mu[j]
    if Var:
        for i in range(1,m+1):
            for j in range(1,m+1):
                A[i,j]=2**.5*(Mu[i]-Mu[j])/(SD[i]**2+SD[j]**2)**.5
    return A

def f(Mu, SD=0, Var = False):
    m=length(Mu)
    A=zeros((m,m), int)
    if not Var:
        for i in range(1,m+1):
            for j in range(1,m+1):
                A[i,j]=pnorm(Mu[i]-Mu[j],0,sqrt(2))

        # diag(A)=1-colSums(A)
  
    if Var:
        for i in range(0,m):
            for j in range(0,m):
                A[i,j]=pnorm(Mu[i]-Mu[j],0,sd=sqrt(SD[i]^2+SD[j]^2))
        # diag(A)=1-colSums(A);
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
    arr = []
    full_arr = []
    for x in range(0, m):
        arr.append(m)
    for y in range(0, m):
        full_arr.append(arr)
    
    # C is the transition matrix, where C[i, j] denotes the number of times that
    C = np.matrix(full_arr) - prior * m * np.zeros((m, m), int).fill_diagonal(1)
  
    pairsDF = pd.DataFrame(Data.pairs)
  
    if pairsDF.shape[1] > 2:
        pairsDF.columns = ["i", "j", "r"]
        C_wide = pairsDF.groupby("i", "j").summarize()
    else:
        pairsDF.columns = ["i", "j"]
        C_wide = pairsDF.groupby("i", "j").summarize()
  
    for l in range(1,C_wide.shape[0]):
        # i wins, j loses
        i = C_wide[l, "i"]
        j = C_wide[l, "j"]
        if C_wide.shape[1] > 2 and weighted: 
            C[i, j] = C[i, j] + C_wide[l, "r"]
        else:
            C[i, j] = C[i, j] + C_wide[l, "n"]

  
    if normalized:
        return normalizeC(C)
    else:
        return C


#' GMM Method for Estimating Random Utility Model wih Normal dsitributions
#' 
#' @param Data.pairs data broken up into pairs
#' @param m number of alternatives
#' @param iter number of iterations to run
#' @param Var indicator for difference variance (default is FALSE)
#' @param prior magnitude of fake observations input into the model
#' @return Estimated mean parameters for distribution of underlying normal (variance is fixed at 1)
#' @export
#' @examples
#' data(Data.Test)
#' Data.Test.pairs <- Breaking(Data.Test, "full")
#' Estimation.Normal.GMM(Data.Test.pairs, 5)
def EstimationNormalGMM(DataPairs, m, iter=1000, Var=False, prior=0):
    
    t0 = time.time() #get starting time
  
    sdhat = ones((1,m), int)
    muhat = ones((1,m), int)
    C = generateC(DataPairs, m, prior)

    if not Var:
        for iter in range(0,iter):
            alpha = 1/iter
            df.sum(axis=1)
            muhat = muhat + alpha *(exp(-delta(muhat)**2/4)*(C - f(muhat))).sum(axis=1)
            muhat = muhat - muhat.min()
        print(((C - f(muhat)).sum() **2)**.5)
  
    if Var:
        for iter in range(0,iter):
            alpha = 1/iter
            muhat = muhat + alpha *(exp(-delta(muhat,sdhat,Var=TRUE)^2/4)*(C - f(muhat,sdhat,Var=TRUE))).sum(axis=1)
            sdhat = abs(sdhat - alpha *rowSums(VarMatrix(sdhat)^(-2)*exp(-delta(muhat,sdhat,Var=TRUE)^2/4)*(C - f(muhat,sdhat,Var=TRUE))))
            muhat = muhat - muhat.min()
            sdhat[1] = 1
        print(sum((C - f(muhat,sdhat,Var=True))^2)^.5)
  
    t = time.time() - t0
  
    params = rep(list(list()), m)
    for i in range(1,m):
        params[[i]]["Mean"] = muhat[1, i]
        params[[i]]["SD"] = sdhat[1, i]

    return list(m = m, order = order(-muhat[1,]), Mean = muhat[1,], SD = sdhat[1,], Time = t, Parameters = params)