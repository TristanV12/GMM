import sys
import time
import numpy as np
import scipy.stats
import plackettluce as pl
#import stats as stats
# import mmgbtl as mm
# import gmmra as gmm
import csv
import glob
import os
from PL_MLE import *


if __name__ == '__main__':
    maxdatasize = 1000
    mm_iters = 100
    mm_epsilon = None
    trialcnt = 0
    rslt_rt_mm = np.zeros((maxdatasize, 10), float)
    rslt_bt_mm = np.zeros((maxdatasize, 10), float)
    rslt_ot_mm = np.zeros((maxdatasize, 10), float)
    for f in glob.glob("data*.csv"):
        trialcnt += 1
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

        # rslt_mse_mm = np.zeros((mm_iters, 10), float)

        # rslt_mm_full = np.zeros((mm_iters, m * 10), float)

        # print("n =   ", end='')
        # sys.stdout.flush()

        # for j in range(0, 10):
        #     n = (j + 1) * 100

        #     alts = [i for i in range(m)]
        #     mmagg = mm.MMPLAggregator(alts)

        #     print("\b"*len(str(j*100)) + str((j+1)*100), end='')
        #     sys.stdout.flush()
        #     votes = np.asarray(data[0:n])
        #     t_mm = time.perf_counter()
        #     gamma_mmfull, btime, otime = mmagg.aggregate(votes, mm_epsilon, mm_iters)
        #     t_mm = time.perf_counter() - t_mm
        #     rslt_mm_full[:, j*10:(j+1)*10 ] = gamma_mmfull
        #     gamma_mm = gamma_mmfull[-1]
        #     rslt_rt_mm[trialcnt-1,j] = t_mm
        #     rslt_bt_mm[trialcnt-1,j] = btime
        #     rslt_ot_mm[trialcnt-1,j] = otime
        #     for itr in range(0, mm_iters):
        #         rslt_mse_mm[itr, j] = stats.mse(gamma, gamma_mmfull[itr])
        for r in range(1, 11):
            data_sample = np.array(data[0:r * 100])
            result = Estimation_PL_MLE(data_sample, mm_iters)

            print("Trial:", trialcnt, "|", r*100, "rankings")
            outname = "output_PL_MLE_10Alternatives_"+str(r*100)+"rankings_"+str(trialcnt)+".csv"
            out = open(outname, "w")
            writer = csv.writer(out)
            writer.writerow(["rankings", r*100])
            writer.writerow(["GT", [ float(x) for x in gt ]])
            for key, value in result.items():
                writer.writerow([key, value])
            out.close()
        filename.close()
        #break
