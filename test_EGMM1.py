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
	for f in glob.glob("MixtureData\\*Mixture.csv"):
		Data = []
		filename = open(f)
		print(filename)
		reader = csv.reader(filename)
		GT1 = next(reader)
		next(reader)
		GT2 = next(reader)
		next(reader)
		alpha = float(next(reader)[0])
		GroudTruth1 = [ float(x) for x in GT1[0].split(' ') ]
		GroudTruth2 = [ float(x) for x in GT2[0].split(' ') ]
		print("GT1", GroudTruth1)
		print("GT2", GroudTruth2)
		print("alpha", alpha)
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
		Data = np.array(Data[0:100])
		DataPairs = Breaking(Data)
		final = EGMM(Data, DataPairs, 6)
		print(final)
		break


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