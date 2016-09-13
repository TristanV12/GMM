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
	itr = 20
	SD = np.array([1]*m)
	files = glob.glob("*Mixture.csv")

	trial = 0
	for file1 in files:
		trial += 1
		print("Running trial", trial)
		print(file1)
		f1 = open(file1, "r")
		reader = csv.reader(f1)
		GT1 = dict(m = m, Mean = list(map(float,next(reader)[0].split(' '))), SD = SD)
		GT2 = dict(m = m, Mean = list(map(float,next(reader)[0].split(' '))), SD = SD)
		alpha = float(next(reader)[0])
		GroundTruth1 = GT1["Mean"]
		GroundTruth2 = GT2["Mean"]

		Data = []
		for iterator in range(2000, n):
			Data.append(list(map(int,next(reader)[0].split(' '))))

		r = 0
		while r < 2000:
			r += 200
			output = open("outputTrial" + str(trial) + "rankings_" + str(r) + ".csv", 'a')
			writer = csv.writer(output, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
			alphas, thetas, cp0, cp1, runtime, se, wse, se1, maxitr = E_GMM(alpha, GroundTruth1, GroundTruth2, Data[0:r], m, k, itr)
			writer.writerow([str(r), "Rankings"])
			writer.writerow(["GT1", GroundTruth1])
			writer.writerow(["GT2", GroundTruth2])
			writer.writerow(["CP1", cp0])
			writer.writerow(["CP2", cp1])
			writer.writerow(["Alpha", alpha])
			writer.writerow(["Mixing_Probabilities", alphas.transpose()])
			writer.writerow(["Optimal_Error", se])
			writer.writerow(["Weighted_Error", wse])
			writer.writerow(["SE1", se1])
			writer.writerow(["Runtime", runtime])
			output.close()
		f1.close()
		# if trial >= 100:
		# 	break