import numpy as np
from GaussianRUMGMM import *

def mixGauRUMobj_mat(x, breaking):
    breaking = np.reshape(breaking, (6, 6))
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
