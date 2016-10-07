import sys
import time
import numpy as np
import scipy.stats
import plackettluce as pl
import stats as stats
import mmgbtl as mm
import gmmpl as gmm
import csv
import glob
import os


if __name__ == '__main__':
    maxdatasize = 1000
    trialcnt = 0
    #Make sure the data are in the following directory
    #TODO: create path if not exist
    for f in glob.glob("data/dataUniform/*.csv"):
        trialcnt += 1
        print("Trial: ", trialcnt)
        filename = open(f)
        reader = csv.reader(filename)
        next(reader)
        gt = next(reader)
        gamma = [ float(x) for x in gt ]
        m = len(gamma)
        gamma = np.asarray(gamma)

        data = []
        for itr in range(0, maxdatasize):
            data.append([ int(x) for x in next(reader)])

        rslt_gmm = np.zeros((11, 14), float)
        rslt_gmm[0, 0:m] = gamma

        print("n =   ", end='')

        for j in range(0, 10):
            n = (j + 1) * 100
            rslt = np.zeros((3, m + 4), float)
            alts = [i for i in range(m)]

            gmmagg = gmm.GMMPLAggregator(alts)

            sys.stdout.flush()
            print("\b"*len(str(j*100)) + str((j+1)*100), end='')
            sys.stdout.flush()
            votes = np.asarray(data[0:n])

            t_gmm = time.perf_counter()
            gamma_gmm, btime, otime = gmmagg.aggregate(votes, k = None) #full breaking. For top 1 breaking, let k = 1
            t_gmm = time.perf_counter() - t_gmm

            mse_gmm = stats.mse(gamma, gamma_gmm) # calc MSE for GMM

            rslt_gmm[j+1, 0:m] = gamma_gmm
            rslt_gmm[j+1, m] = mse_gmm
            rslt_gmm[j+1, m+1] = t_gmm
            rslt_gmm[j+1, m+2] = btime
            rslt_gmm[j+1, m+3] = otime

        print()
        #make sure the following path exists before run this code
        #TODO creat directory if not exists
        outnameGMM = "outputGMM/rslt_gmm_"+str(trialcnt)+".csv"
        np.savetxt(outnameGMM, rslt_gmm, delimiter=',', newline="\r\n")
        #break
