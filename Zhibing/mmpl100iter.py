import sys
import time
import numpy as np
import scipy.stats
import plackettluce as pl
import stats as stats
import mmpl as mm
import csv
import glob
import os


if __name__ == '__main__':
    maxdatasize = 1000
    mm_iters = 100
    mm_epsilon = None
    trialcnt = 0
    rslt_rt_mm = np.zeros((maxdatasize, 10), float)

    #make sure data are in this directory
    #TODO raise an error if directory does not exist

    for f in glob.glob("data*.csv"):
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

        rslt_mse_mm = np.zeros((mm_iters, 10), float)

        rslt_mm_full = np.zeros((mm_iters, m * 10), float)

        print("n =   ", end='')
        sys.stdout.flush()

        for j in range(0, 10):
            n = (j + 1) * 100

            alts = [i for i in range(m)]
            mmagg = mm.MM_PL(alts)

            print("\b"*len(str(j*100)) + str((j+1)*100), end='')
            sys.stdout.flush()
            votes = np.asarray(data[0:n])
            t_mm = time.perf_counter()
            gamma_mmfull = mmagg.aggregate(votes, mm_iters)
            t_mm = time.perf_counter() - t_mm
            rslt_mm_full[:, j*100:(j+1)*100 ] = gamma_mmfull
            gamma_mm = gamma_mmfull[-1]
            rslt_rt_mm[trialcnt-1,j] = t_mm
            for itr in range(0, mm_iters):
                rslt_mse_mm[itr, j] = stats.mse(gamma, gamma_mmfull[itr])

        print()
        #make sure these directories exist
        #TODO create directories if not exist
        outnameMM_mse = "rslt_mmpl_mse_"+str(trialcnt)+".csv"
        outnameMMfull = "rslt_mmpl_est_"+str(trialcnt)+".csv"
        np.savetxt(outnameMM_mse, rslt_mse_mm, delimiter=',', newline="\r\n")
        np.savetxt(outnameMMfull, rslt_mm_full, delimiter=',', newline="\r\n")
    np.savetxt("mmpl_rt.csv", rslt_rt_mm, delimiter=',', newline="\r\n")

        #break
