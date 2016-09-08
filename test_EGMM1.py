from generate import *
from EstimationNormalGMM import *
from likelihood_rum import *
from EGMM import *
from GaussianRUMGMM import *
import csv
import sys
import glob
import os

if __name__ == '__main__':
	#for f in glob.glob("MixturesGT1to10/*Mixture.csv"):
	for f in glob.glob("testData/*Mixture.csv"):
		m = 6
		n = 1000
		k = 2
		itr = 30
		Data = []
		filename = open(f)
		print(filename)
		reader = csv.reader(filename)
		GT1 = next(reader)
		next(reader)
		GT2 = next(reader)
		next(reader)
		alpha = float(next(reader)[0])
		GroundTruth1 = [ float(x) for x in GT1[0].split(' ') ]
		GroundTruth2 = [ float(x) for x in GT2[0].split(' ') ]
		for i in range(0, m):
			GroundTruth1[i] -= min(GroundTruth1)
			GroundTruth2[i] -= min(GroundTruth2)
		GroundTruth1 = np.asarray(GroundTruth1)
		GroundTruth2 = np.asarray(GroundTruth2)
		print("alpha: ", alpha)
		for line in reader:
			if len(line) == 0:
				try:
					tmp = next(reader)
					if len(tmp) == 0:
						break
					Data.append([ int(x) for x in tmp[0].split(' ') ])
				except StopIteration:
					break
			else:
				Data.append([ int(x) for x in line[0].split(' ') ])
		Data = np.array(Data[0:n])
		#print("Rankings: ", Data)
		#print("Breaking: ", DataPairs)
		alphas, thetas, cp0, cp1 = E_GMM(alpha, GroundTruth1, GroundTruth2, Data, m, k, itr)
		print("mixing probabilites: ", alphas.transpose())
		print("CP0: ", cp0)
		print("CP1: ", cp1)
		print("Alpha: ", alpha)
		print("GT0: ", GroundTruth1)
		print("GT1: ", GroundTruth2)
		#weights = np.ones((1, n), float)
		#breaking = dataBreaking(Data, m, 1, weights)
		#print("Single RUM: ", GMMGaussianRUM(breaking[0], m, n, itr=2))
		break

		#0.898 105243 521304
		#0.496 521304 142503
	# print("ground truth", Params)
	# print("\n\nGround truth order", Params["order"])
	# #print(likelihoodRUM(Data,Params, "norm"))
	# DataPairs = Breaking(Data)
	# #print("\n\nData", Data)
	# final = EGMM(Data, DataPairs, 6)
	# print(final)
	# mu = Params["Mean"]
	# mu = mu - mu.min()
	# muhat = final[0]["Mean"]
	# muhat = muhat - muhat.min()
	# mse = 0
	# for itr in range(0,len(muhat)):
	# 	mse += (muhat[itr] - mu[itr]) ** 2
	# print("MSE:", mse)
