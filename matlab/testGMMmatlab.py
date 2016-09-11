import numpy as np
from GaussianRUMGMM import *
from mixGaussianRUMGMM import *
from scipy.optimize import fmin_cg, minimize
from EGMM import *
from generate import *

if __name__ == '__main__':

    m = 6
    n = 10000
    k = 2
    itr = 1
    GT1 = GenerateRUMParameters(m, "normal")
    GT2 = GenerateRUMParameters(m, "normal")
    GroundTruth1 = GT1["Mean"]
    GroundTruth2 = GT2["Mean"]
    alpha = np.random.rand()
    tBreaking = alpha * trueBreaking(GroundTruth1) + (1 - alpha) * trueBreaking(GroundTruth2)
    dataset1 = GenerateRUMData(GT1, m, n, "normal")
    dataset2 = GenerateRUMData(GT2, m, n, "normal")
    print("Ground Truth: ", alpha)
    print(GroundTruth1)
    print(GroundTruth2)
    Data = []
    y = np.random.rand(1, n)[0]
    cnt1 = 0
    cnt2 = 0
    for i in range(n):
        if y[i] <= alpha:
            Data.append(dataset1[cnt1])
            cnt1 += 1
        else:
            Data.append(dataset2[cnt2])
            cnt2 += 1
    DataDict = Dictionarize(Data)
    Breaking = dataBreaking(DataDict, m)
    x = np.append(GroundTruth1, GroundTruth2)
    x = np.insert(x, 0, alpha)
    #print(x)
    #print("Breaking Diff: ", Breaking - n*tBreaking)
    print("ObjFuncValueAtTrueParamsTrueBreaking: ", mixGauRUMobj(x, tBreaking))
    print("ObjFuncValueAtTrueParamsWithData: ", mixGauRUMobj(x, Breaking))
    for i in range(0, itr):
        rslt = mixGauRUM_matlab(Breaking)
        trslt = mixGauRUM_matlab(Breaking)
        print("True breaking result: ", trslt[0])
        print(trslt[1:7])
        print(trslt[7:])
    print("Number of Trials: ", i + 1)
    print("ObjFuncValueAtEstParamsMinimize: ", mixGauRUMobj(rslt, Breaking))
    se, cp0, cp1 = se2Mix(alpha, GroundTruth1, GroundTruth2, rslt[0], rslt[1:7], rslt[7:13], optimize = True, weighted = False, normalize = False)
    wse, cp0, cp1 = se2Mix(alpha, GroundTruth1, GroundTruth2, rslt[0], rslt[1:7], rslt[7:13], optimize = True, weighted = True, normalize = False)
    print("SE: ", se)
    print("WSE: ", wse)
    print("Minimize result: ", rslt[0])
    print("cp0:", cp0)
    print("cp1:", cp1)
