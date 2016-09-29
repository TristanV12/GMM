from generate import *
#from EstimationNormalGMM import *
from likelihood_rum import *
from EGMM import *
from GaussianRUMGMM import *
import csv
import sys
import glob
import os
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import stats

if __name__ == '__main__':
	m = 10
	files = glob.glob("PL_MLE_Trials\\*.csv")

	trial = 0
	print(len(files))

	mse = []
	time = [0] * 10
	for i in range(0, 10):
		inner = [0]*100
		mse.append(inner)
	for file1 in files:
		trial += 1
		if trial % 1000 == 0:
			print("Running trial", trial)

		#open file for reading
		f1 = open(file1, "r")
		reader = csv.reader(f1)
		rankings = int(next(reader)[1])

		#get ground truth
		gt_string = next(reader)[1]
		gt = []
		for g in gt_string.split(' '):
			g_new = g.replace(',', '')
			g_new = g_new.replace('[', '')
			g_new = g_new.replace(']', '')
			gt.append(float(g_new))

		#get mses
		for line in reader:
			tpe = line[0]
			if tpe == "Time":
				time[(int(rankings / 100) - 1)] += float(line[1]) / 100
			if tpe == "Mean":
				count = 0
				for x in line[1].split("]"):
					y = x.replace('[',' ')
					y = y.replace('\n',' ')
					z = y.split(" ")
					inner = []
					for w in z:
						if w != '':
							inner.append(float(w))
					if len(inner) != 0:
						inner = stats.mse(gt, inner)
						#print(count, inner)
						mse[int((rankings / 100) - 1)][count] += inner / 100
					count += 1
		f1.close()

	#print
	for l in range(0, len(mse)):
		plt.plot(list(range(0, 100)), np.array(mse[l]), marker="o", color='red', linewidth=4, label='Average WMSE')
		# plt.ylabel("MSE " + str((l + 1) * 100) + " rankings")
		# plt.xlabel("Iterations")
		# plt.show()
		print("\nMSE " + str((l + 1) * 100) + " rankings")
		for m in range(0, len(mse[l])):
			print("Iteration", str(m + 1) + ":", mse[l][m])
	# plt.plot(list(range(1, 11)), np.array([mse[0][-1],mse[1][-1],mse[2][-1],mse[3][-1],mse[4][-1],mse[5][-1],mse[6][-1],mse[7][-1],mse[8][-1],mse[9][-1]]), marker="o", color='red', linewidth=4, label='Average WMSE')
	# plt.ylabel("MSE")
	# plt.xlabel("Number of rankings used / 100")
	# plt.show()
	# plt.plot(list(range(1, 11)), np.array(time), marker="o", color='red', linewidth=4, label='Average WMSE')
	# plt.ylabel("Runtime")
	# plt.xlabel("Number of rankings used / 100")
	# plt.show()













	# for q in range(0,10):
	# 	ave_MSE[q] = ave_MSE[q] / countArray[q]
	# 	ave_WMSE[q] = ave_WMSE[q] / countArray[q]
	# 	ave_time[q] = ave_time[q] / countArray[q]
	# 	print("For", str((q + 1)*200), "rankings:")
	# 	print("Average MSE Plackett Luce Sense:", ave_MSE[q])
	# 	print("Average Weighted Error:", ave_WMSE[q])
	# 	print("Average Runtime:", ave_time[q], "\n\n")
	# 	for w in range(0, num_datapoints):
	# 		if alpha_tot[q][w] != 0:
	# 			mses_alpha[q][w] = mses_alpha[q][w] / alpha_tot[q][w]
	# 			wmses_alpha[q][w] = wmses_alpha[q][w] / alpha_tot[q][w]
	# 			wmseA[q][w] = wmseA[q][w] / alpha_tot[q][w]
	# 			wmseB[q][w] = wmseB[q][w] / alpha_tot[q][w]
	# 			alphaA[q][w] = alphaA[q][w] / alpha_tot[q][w]
		
	# 	plt.plot(alpha_dist, np.array(wmses_alpha[q]), marker="o", color='orange', linewidth=4, label='MWSE')
	# 	plt.ylabel("MWSE")
	# 	plt.xlabel("Alpha")
	# 	plt.show()
	# 	# plt.plot(alpha_dist, np.array(wmseA[q]), marker="o", color='red', linewidth=4, label='Average WMSE')
	# 	# plt.plot(alpha_dist, np.array(wmseB[q]), marker="v", color='blue', linewidth=4, label='Average WMSE')
	# 	# plt.plot(alpha_dist, np.array(alphaA[q]), marker=".", color='green', linewidth=4, label='Average WMSE')
	# 	# plt.ylabel("Average WMSE")
	# 	# plt.xlabel("Alpha")
	# 	# plt.show()
	# print(countArray)
	# print(ave_MSE)
	# plt.plot(np.array(x), np.array(ave_MSE), color='red', linewidth=4, label='Average Optimum MSE')
	# plt.ylabel("Average MSE Plackett Luce Sense")
	# plt.xlabel("# of Rankings / 100")
	# plt.show()
	# print(ave_WMSE)
	# plt.plot(np.array(x), np.array(ave_WMSE), color='blue', linewidth=4, label='Average Weighted MSE')
	# plt.ylabel("MWSE")
	# plt.xlabel("# of Rankings / 100")
	# plt.show()
	# print(ave_time)
	# plt.plot(np.array(x), np.array(ave_time), color='green', linewidth=4, label='Average Runtime')
	# plt.ylabel("Average Runtime")
	# plt.xlabel("# of Rankings / 100")
	# plt.show()