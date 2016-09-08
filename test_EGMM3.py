from generate import *
#from EstimationNormalGMM import *
from likelihood_rum import *
from EGMM import *
from GaussianRUMGMM import *
import csv
import sys
import glob
import os

if __name__ == '__main__':
	m = 6
	n = 10000
	k = 2
	itr = 50
	SD = np.array([1]*m)
	files = glob.glob("*Mixture.csv")

	trial = 0
	for file1 in files:
		trial += 1
		print("Running trial", trial)
		f1 = open(file1, "r")
		output = open("outputTrial" + str(trial) + ".csv", 'a')
		reader = csv.reader(f1)
		writer = csv.writer(output, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		GT1 = dict(m = m, Mean = list(map(float,next(reader)[0].split(' '))), SD = SD)
		next(reader)
		GT2 = dict(m = m, Mean = list(map(float,next(reader)[0].split(' '))), SD = SD)
		next(reader)
		alpha = float(next(reader)[0])
		GroundTruth1 = GT1["Mean"]
		GroundTruth2 = GT2["Mean"]

		Data = []
		for itr in range(0, n):
			next(reader)
			Data.append(list(map(int,next(reader)[0].split(' '))))
		alphas, thetas, cp0, cp1, runtime, se, wse, se1 = E_GMM(alpha, GroundTruth1, GroundTruth2, Data[0:100], m, k, itr)
		writer.writerow(["100 Rankings"])
		writer.writerow(["GT1", GroundTruth1])
		writer.writerow(["GT2", GroundTruth2])
		writer.writerow(["Alpha", alpha])
		writer.writerow(["Optimal Error", se])
		writer.writerow(["Weighted Error", wse])
		writer.writerow(["SE1", se1])
		writer.writerow(["Runtime", runtime])
		alphas, thetas, cp0, cp1, runtime, se, wse, se1 = E_GMM(alpha, GroundTruth1, GroundTruth2, Data[0:1000], m, k, itr)
		writer.writerow(["1000 Rankings"])
		writer.writerow(["GT1", GroundTruth1])
		writer.writerow(["GT2", GroundTruth2])
		writer.writerow(["Alpha", alpha])
		writer.writerow(["Optimal Error", se])
		writer.writerow(["Weighted Error", wse])
		writer.writerow(["SE1", se1])
		writer.writerow(["Runtime", runtime])
		alphas, thetas, cp0, cp1, runtime, se, wse, se1 = E_GMM(alpha, GroundTruth1, GroundTruth2, Data[0:10000], m, k, itr)
		writer.writerow(["10000 Rankings"])
		writer.writerow(["GT1", GroundTruth1])
		writer.writerow(["GT2", GroundTruth2])
		writer.writerow(["Alpha", alpha])
		writer.writerow(["Optimal Error", se])
		writer.writerow(["Weighted Error", wse])
		writer.writerow(["SE1", se1])
		writer.writerow(["Runtime", runtime])
		f1.close

	# alphas, thetas, cp0, cp1 = E_GMM(alpha, GroundTruth1, GroundTruth2, Data, m, k, itr)
	# print("mixing probabilites: ", alphas.transpose())
	# print("CP0: ", cp0)
	# print("CP1: ", cp1)
	# print("Alpha: ", alpha)
	# print("GT0: ", GroundTruth1)
	# print("GT1: ", GroundTruth2)
