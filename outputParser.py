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

def normMeans(theta, PL = True):
	if PL:
		thetap = np.exp(theta)
		return thetap/np.sum(thetap)
	else:
		return theta - theta.min()

#If output optimal error, set optimize = True
#If output PL sense error, set optimize = False, PL = True, and set weighted = True or False depending on needs.
def se2Mix(alpha, gt1, gt2, hatalpha, cp1, cp2, optimize = True, weighted = True, normalize = False, PL = False):
	if optimize:
		m = len(gt1)
		d11 = (np.sum(gt1) - np.sum(cp1))/m
		d12 = (np.sum(gt2) - np.sum(cp2))/m
		d21 = (np.sum(gt1) - np.sum(cp2))/m
		d22 = (np.sum(gt2) - np.sum(cp1))/m
		if weighted:
			part11 = np.sum((gt1 - cp1 - d11) ** 2)
			part12 = np.sum((gt2 - cp2 - d12) ** 2)
			part13 = (alpha - hatalpha) ** 2
			part21 = np.sum((gt1 - cp2 - d21) ** 2)
			part22 = np.sum((gt2 - cp1 - d22) ** 2)
			part23 = (alpha + hatalpha - 1) ** 2
			se1 = (alpha * part11 + (1 - alpha) * part12 + part13)
			se2 = ((1 - alpha) * part21 + alpha * part22 + part23)
		else:
			se1 = np.sum((gt1 - cp1 - d11) ** 2) + np.sum((gt2 - cp2 - d12) ** 2) + (alpha - hatalpha) ** 2
			se2 = np.sum((gt1 - cp2 - d21) ** 2) + np.sum((gt2 - cp1 - d22) ** 2) + (alpha + hatalpha - 1) ** 2
		if se1 <= se2:
			return se1, part11, part12, part13
		else:
			return se2, part21, part22, part23
	else:
		if PL:
			gt1 = normMeans(gt1, PL = True)
			gt2 = normMeans(gt2, PL = True)
			cp1 = normMeans(cp1, PL = True)
			cp2 = normMeans(cp2, PL = True)
		elif normalize:
			gt1 -= np.log(np.sum(np.exp(gt1)))
			gt2 -= np.log(np.sum(np.exp(gt2)))
			cp1 -= np.log(np.sum(np.exp(cp1)))
			cp2 -= np.log(np.sum(np.exp(cp2)))
		else:
			gt1 = normMeans(gt1, PL = False)
			gt2 = normMeans(gt2, PL = False)
			cp1 = normMeans(cp1, PL = False)
			cp2 = normMeans(cp2, PL = False)
		if weighted:
			se1 = alpha * np.sum((gt1 - cp1) ** 2) + (1 - alpha) * np.sum((gt2 - cp2) ** 2) + (alpha - hatalpha) ** 2
			se2 = (1 - alpha) * np.sum((gt1 - cp2) ** 2) + alpha * np.sum((gt2 - cp1) ** 2) + (alpha + hatalpha - 1) ** 2
		else:
			se1 = np.sum((gt1 - cp1) ** 2) + np.sum((gt2 - cp2) ** 2) + (alpha - hatalpha) ** 2
			se2 = np.sum((gt1 - cp2) ** 2) + np.sum((gt2 - cp1) ** 2) + (alpha + hatalpha - 1) ** 2
		return min(se1, se2)#, cp1, cp2


if __name__ == '__main__':
	m = 6
	n = 10000
	k = 2
	itr = 20
	SD = np.array([1]*m)
	ave_MSE = [0]*10
	ave_WMSE = [0]*10
	ave_SE1 = [0]*10
	ave_time = [0]*10
	array = [[], [], [], [], [], [], [], [], [], []]
	alphas = []
	num_datapoints = 11
	alpha_dist = np.linspace(0.0, 1.0, num=num_datapoints)
	alpha_tot = [[0]*num_datapoints, [0]*num_datapoints, [0]*num_datapoints, [0]*num_datapoints,
		[0]*num_datapoints, [0]*num_datapoints, [0]*num_datapoints, [0]*num_datapoints, [0]*num_datapoints,
		[0]*num_datapoints]
	mses_alpha = [[0]*num_datapoints, [0]*num_datapoints, [0]*num_datapoints, [0]*num_datapoints,
		[0]*num_datapoints, [0]*num_datapoints, [0]*num_datapoints, [0]*num_datapoints, [0]*num_datapoints,
		[0]*num_datapoints]
	wmses_alpha = [[0]*num_datapoints, [0]*num_datapoints, [0]*num_datapoints, [0]*num_datapoints,
		[0]*num_datapoints, [0]*num_datapoints, [0]*num_datapoints, [0]*num_datapoints, [0]*num_datapoints,
		[0]*num_datapoints]
	wmseB = [[0]*num_datapoints, [0]*num_datapoints, [0]*num_datapoints, [0]*num_datapoints,
		[0]*num_datapoints, [0]*num_datapoints, [0]*num_datapoints, [0]*num_datapoints, [0]*num_datapoints,
		[0]*num_datapoints]
	wmseA = [[0]*num_datapoints, [0]*num_datapoints, [0]*num_datapoints, [0]*num_datapoints,
		[0]*num_datapoints, [0]*num_datapoints, [0]*num_datapoints, [0]*num_datapoints, [0]*num_datapoints,
		[0]*num_datapoints]
	alphaA = [[0]*num_datapoints, [0]*num_datapoints, [0]*num_datapoints, [0]*num_datapoints,
		[0]*num_datapoints, [0]*num_datapoints, [0]*num_datapoints, [0]*num_datapoints, [0]*num_datapoints,
		[0]*num_datapoints]
	countArray = [0] * 10
	x = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
	files = (glob.glob("output20ItrFixed\\output3\\*.csv") + glob.glob("output20ItrFixed\\output9-25-2016\\*.csv") + glob.glob("output20ItrFixed\\output20Itr_1\\*.csv"))
		# + glob.glob("output20Itr\\output3\\*.csv") + glob.glob("output20Itr\\output4\\*.csv")
		# + glob.glob("output20Itr\\output5\\*.csv") + glob.glob("output20Itr\\output6\\*.csv")
		# + glob.glob("output20Itr\\output7\\*.csv"))

	trial = 0
	print(len(files))
	for file1 in files:
		trial += 1
		if trial % 1000 == 0:
			print("Running trial", trial)
		f1 = open(file1, "r")
		reader = csv.reader(f1)
		ranks = int((int(next(reader)[0].split(' ')[0])) / 200) - 1
		gt1 = next(reader)
		for g in range(0, len(gt1)):
			gt1[g] = float(gt1[g].replace("|[", "").replace("]|", "").replace(" ", "").replace("GT1", ""))
		gt2 = next(reader)
		for g in range(0, len(gt2)):
			gt2[g] = float(gt2[g].replace("|[", "").replace("]|", "").replace(" ", "").replace("GT2", ""))
		cp1 = []
		cp = next(reader)[0]
		cp = cp.replace("|[", "").replace("]|", "").replace("CP1", "").split(' ')
		for c in cp:
			if c != '':
				cp1.append(float(c))
		cp2 = []
		cp = next(reader)[0]
		cp = cp.replace("|[", "").replace("]|", "").replace("CP2", "").split(' ')
		for c in cp:
			if c != '':
				cp2.append(float(c))
		if len(cp2) != 6:
			temp = cp1 + cp2
			cp1 = temp[:6]
			cp2 = temp[6:]
			cp = next(reader)[0]
			cp = cp.replace("|[", "").replace("]|", "").replace("CP2", "").split(' ')
			for c in cp:
				if c != '':
					cp2.append(float(c))
		if len(cp1) != 6 or len(cp2) != 6 or len(gt1) != 6 or len(gt2) != 6:
			print(cp1)
			print(cp2)
			print(gt1)
			print(gt2)
			print(file1)
			print("Fail")
		# for g in range(0, len(cp2)):
		# 	cp2[g] = float(cp2[g].replace("|[", "").replace("]|", "").replace(" ", "").replace("CP2", ""))
		alpha = float(next(reader)[0].split(' ')[1])
		

		hatalpha = float(next(reader)[0].replace("Mixing_Probabilities", "").replace("|[[", "").replace("]]|", "").strip().split(' ')[0])
		#print(hatalpha)
		

		arr = next(reader)[0].split(' ')
		#print("arr", arr
		if(len(arr) != 2):
			arr = next(reader)[0].split(' ')
		omse = float(arr[1])
		wmse = float(next(reader)[0].split(' ')[1])
		se1 = float(next(reader)[0].split(' ')[1])
		time = float(next(reader)[0].split(' ')[1])


		if alpha > 0 and alpha < 1:
			mse_pl = se2Mix(alpha, np.array(gt1), np.array(gt2), hatalpha, np.array(cp1),
				np.array(cp2), optimize=False, weighted=False, PL=True)
			wmse_real, p1, p2, p3 = se2Mix(alpha, np.array(gt1), np.array(gt2), hatalpha, np.array(cp1),
				np.array(cp2), optimize=True, weighted=True, PL=False)
			ave_MSE[ranks] += mse_pl
			ave_WMSE[ranks] += wmse_real
			ave_time[ranks] += time
			countArray[ranks] += 1

			tmp = round(alpha * (num_datapoints - 1))
			alpha_tot[ranks][tmp] += 1
			mses_alpha[ranks][tmp] += mse_pl
			wmses_alpha[ranks][tmp] += wmse_real
			wmseA[ranks][tmp] += p1
			wmseB[ranks][tmp] += p2
			alphaA[ranks][tmp] += p3
		# if omse > 2:
		# 	array[ranks].append(alpha)
		#array[ranks].append(omse)
		# if omse > 1 and ranks == 0:
		# 	print(omse)
		f1.close()
	# array[0] = sorted(array[0])
	# array[1] = sorted(array[1])
	# array[2] = sorted(array[2])
	# array[3] = sorted(array[3])
	# array[4] = sorted(array[4])
	# a = np.array(array[0])
	# b = np.array(array[1])
	# c = np.array(array[2])
	# d = np.array(array[3])
	# e = np.array(array[4])
	# # for q in range(0, len(array[0])):
	# # 	if array[0][q] > 2:
	# # 		a = np.array(array[0][:q])
	# # 		break
	# # for q in range(0, len(array[1])):
	# # 	if array[1][q] > 2:
	# # 		b = np.array(array[1][:q])
	# # 		break
	# # for q in range(0, len(array[2])):
	# # 	if array[2][q] > 2:
	# # 		c = np.array(array[2][:q])
	# # 		break
	# # for q in range(0, len(array[3])):
	# # 	if array[3][q] > 2:
	# # 		q0 = q
	# # 		d = np.array(array[3][:q])
	# # 		break
	# # for q in range(0, len(array[4])):
	# # 	if array[4][q] > 2:
	# # 		q0 = q
	# # 		e = np.array(array[4][:q])
	# # 		break
	# # print(len(array[0]), len(array[1]), len(array[2]), len(array[3]), len(array[4]))
	# # print(a)
	# hfont = {'fontname':'Helvetica'}
	# plt.hist(a, bins=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])
	# plt.rcParams['xtick.labelsize'] = 20
	# plt.rcParams['ytick.labelsize'] = 20
	# plt.ylabel("Frequency")
	# plt.xlabel("Alpha where MSE > 2 n=200")
	# plt.show()
	# hfont = {'fontname':'Helvetica'}
	# plt.hist(b, bins=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])
	# plt.rcParams['xtick.labelsize'] = 20
	# plt.rcParams['ytick.labelsize'] = 20
	# plt.ylabel("Frequency")
	# plt.xlabel("Alpha where MSE > 2 n=400")
	# plt.show()
	# hfont = {'fontname':'Helvetica'}
	# plt.hist(c, bins=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])
	# plt.rcParams['xtick.labelsize'] = 20
	# plt.rcParams['ytick.labelsize'] = 20
	# plt.ylabel("Frequency")
	# plt.xlabel("Alpha where MSE > 2 n=600")
	# plt.show()
	# hfont = {'fontname':'Helvetica'}
	# plt.hist(d, bins=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])
	# plt.rcParams['xtick.labelsize'] = 20
	# plt.rcParams['ytick.labelsize'] = 20
	# plt.ylabel("Frequency")
	# plt.xlabel("Alpha where MSE > 2 n=800")
	# plt.show()
	# hfont = {'fontname':'Helvetica'}
	# plt.hist(e, bins=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])
	# plt.rcParams['xtick.labelsize'] = 20
	# plt.rcParams['ytick.labelsize'] = 20
	# plt.ylabel("Frequency")
	# plt.xlabel("Alpha where MSE > 2 n=1000")
	# plt.show()
	for q in range(0,10):
		ave_MSE[q] = ave_MSE[q] / countArray[q]
		ave_WMSE[q] = ave_WMSE[q] / countArray[q]
		ave_time[q] = ave_time[q] / countArray[q]
		print("For", str((q + 1)*200), "rankings:")
		print("Average MSE Plackett Luce Sense:", ave_MSE[q])
		print("Average Weighted Error:", ave_WMSE[q])
		print("Average Runtime:", ave_time[q], "\n\n")
		for w in range(0, num_datapoints):
			if alpha_tot[q][w] != 0:
				mses_alpha[q][w] = mses_alpha[q][w] / alpha_tot[q][w]
				wmses_alpha[q][w] = wmses_alpha[q][w] / alpha_tot[q][w]
				wmseA[q][w] = wmseA[q][w] / alpha_tot[q][w]
				wmseB[q][w] = wmseB[q][w] / alpha_tot[q][w]
				alphaA[q][w] = alphaA[q][w] / alpha_tot[q][w]
		
		plt.plot(alpha_dist, np.array(wmses_alpha[q]), marker="o", color='orange', linewidth=4, label='MWSE')
		plt.ylabel("MWSE")
		plt.xlabel("Alpha")
		plt.show()
		# plt.plot(alpha_dist, np.array(wmseA[q]), marker="o", color='red', linewidth=4, label='Average WMSE')
		# plt.plot(alpha_dist, np.array(wmseB[q]), marker="v", color='blue', linewidth=4, label='Average WMSE')
		# plt.plot(alpha_dist, np.array(alphaA[q]), marker=".", color='green', linewidth=4, label='Average WMSE')
		# plt.ylabel("Average WMSE")
		# plt.xlabel("Alpha")
		# plt.show()
	print(countArray)
	print(ave_MSE)
	plt.plot(np.array(x), np.array(ave_MSE), color='red', linewidth=4, label='Average Optimum MSE')
	plt.ylabel("Average MSE Plackett Luce Sense")
	plt.xlabel("# of Rankings / 100")
	plt.show()
	print(ave_WMSE)
	plt.plot(np.array(x), np.array(ave_WMSE), color='blue', linewidth=4, label='Average Weighted MSE')
	plt.ylabel("MWSE")
	plt.xlabel("# of Rankings / 100")
	plt.show()
	print(ave_time)
	plt.plot(np.array(x), np.array(ave_time), color='green', linewidth=4, label='Average Runtime')
	plt.ylabel("Average Runtime")
	plt.xlabel("# of Rankings / 100")
	plt.show()