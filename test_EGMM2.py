from generate import *
#from EstimationNormalGMM import *
from likelihood_rum import *
from EGMM import *
from GaussianRUMGMM import *
#import csv
#import sys
#import glob
#import os

if __name__ == '__main__':
	m = 6
	n = 2000
	k = 2
	itr = 50
	alpha = np.random.rand()
	print(alpha)
	Data = []
	GT1 = GenerateRUMParameters(m, "normal")
	GT2 = GenerateRUMParameters(m, "normal")
	dataset1 = GenerateRUMData(GT1, m, n, "normal")
	dataset2 = GenerateRUMData(GT2, m, n, "normal")

	x = np.random.rand(1, n)[0]
	cnt1 = 0
	cnt2 = 0
	for i in range(n):
		if x[i] <= alpha:
			Data.append(dataset1[cnt1])
			cnt1 += 1
		else:
			Data.append(dataset2[cnt2])
			cnt2 += 1
	GroundTruth1 = GT1["Mean"]
	GroundTruth2 = GT2["Mean"]
	alphas, thetas, cp0, cp1 = E_GMM(alpha, GroundTruth1, GroundTruth2, Data, m, k, itr)
	print("mixing probabilites: ", alphas.transpose())
	print("CP0: ", cp0)
	print("CP1: ", cp1)
	print("Alpha: ", alpha)
	print("GT0: ", GroundTruth1)
	print("GT1: ", GroundTruth2)
