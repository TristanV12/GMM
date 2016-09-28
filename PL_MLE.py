import time        # used to get runtime
import numpy as np # used for matrices and vectors

def Estimation_PL_MLE(Data, iter = 10):
    rank = Data
    t0 = time.time()
    m = len(rank[0]) # number of alternatives
    n = len(rank)    # number of agents

    GammaTotal = np.zeros((iter,m))
    mean = np.zeros((iter,m))
    LTotal = np.zeros((iter,))
  
    M = np.zeros((1,n), dtype=np.int)
    for i in range(0, n):
        if sum(rank[i]) == 0:
            M[0, i] = rank[i].index(0) - 1
        else:
            M[0, i] = m

    W = np.zeros((1,m))
    for t in range(0, m):
        for j in range(0, n):
            W[0,t] = (rank[j][M[0,j] - 1] == t + 1) + W[0,t]
        W[0,t] = n - W[0,t]
    gamma = np.ones((1,m))

    for itr in range(0, iter):
        gamtemp = gamma

        for t in range(0, m):
            denom = .1
            for j in range(0, n):
                for i in range(0, (M[0, j] - 1)):
                    delta = sum(rank[j,i:M[0, j]] == t + 1)
                    denomt3 = 0
                    for s in range(i, M[0, j]):
                        denomt3 = denomt3 + gamma[0, rank[j,s] - 1]
                    denom = (delta / denomt3) + denom
            gamtemp[0, t] = W[0, t] / denom
        gamma = gamtemp
        GammaTotal[itr] = gamma/sum(gamma[0])

        ll = 0

        for i in range(0, n):
            mj = sum(rank[i] > 0)
            temp = gamma[0, rank[i][rank[i] > 0] - 1]
            temp = temp[0:mj]
            temp = temp[::-1]
            ll = ll + sum(np.log(temp)) - sum(np.log(np.cumsum(temp)))
        LTotal[itr] = ll

        print("Iteration", itr)
    for i in range(0,iter):
        mean[i] = (1 / GammaTotal[i]) / sum(1 / GammaTotal[i])

    t = time.time() - t0

    return dict(m=m,
        order=mean.ravel().argsort(),
        Mean=mean,
        SD=mean,
        LL=LTotal,
        Time=t)#,
        #AverageLogLikelihood=t(LTotal[iter - 1]/n))