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


def se2Mix(alpha, gt1, gt2, hatalpha, cp1, cp2, optimize = True, weighted = True, normalize = False):
	if optimize:
		m = len(gt1)
		d11 = (np.sum(gt1) - np.sum(cp1))/m
		d12 = (np.sum(gt2) - np.sum(cp2))/m
		d21 = (np.sum(gt1) - np.sum(cp2))/m
		d22 = (np.sum(gt2) - np.sum(cp1))/m
		if weighted:
			se1 = alpha * np.sum((gt1 - cp1 - d11) ** 2) + (1 - alpha) * np.sum((gt2 - cp2 - d12) ** 2) + (alpha - hatalpha) ** 2
			se2 = (1 - alpha) * np.sum((gt1 - cp2 - d21) ** 2) + alpha * np.sum((gt2 - cp1 - d22) ** 2) + (alpha + hatalpha - 1) ** 2
		else:
			se1 = np.sum((gt1 - cp1 - d11) ** 2) + np.sum((gt2 - cp2 - d12) ** 2) + (alpha - hatalpha) ** 2
			se2 = np.sum((gt1 - cp2 - d21) ** 2) + np.sum((gt2 - cp1 - d22) ** 2) + (alpha + hatalpha - 1) ** 2
		return min(se1, se2)
	else:
		if normalize:
			gt1 -= np.log(np.sum(np.exp(gt1)))
			gt2 -= np.log(np.sum(np.exp(gt2)))
			cp1 -= np.log(np.sum(np.exp(cp1)))
			cp2 -= np.log(np.sum(np.exp(cp2)))
		else:
			gt1 -= gt1.min()
			gt2 -= gt2.min()
			cp1 -= cp1.min()
			cp2 -= cp2.min()
		se1 = np.sum((gt1 - cp1) ** 2) + np.sum((gt2 - cp2) ** 2) + (alpha - hatalpha) ** 2
		se2 = np.sum((gt1 - cp2) ** 2) + np.sum((gt2 - cp1) ** 2) + (alpha + hatalpha - 1) ** 2
		return min(se1, se2)


if __name__ == '__main__':
	m = 6
	n = 10000
	k = 2
	itr = 20
	SD = np.array([1]*m)
	ave_OMSE = [0]*5
	ave_WMSE = [0]*5
	ave_SE1 = [0]*5
	ave_time = [0]*5
	array = [[], [], [], [], []]
	alphas = []
	countArray = [0, 0, 0, 0, 0]
	x = [200, 400, 600, 800, 1000]
	files = glob.glob("output\\*.csv") + glob.glob("output2\\*.csv") + glob.glob("output3\\*.csv")

	trial = 0
	#print(len(files))
	for file1 in files:
		trial += 1
		#print("Running trial", trial)
		f1 = open(file1, "r")
		reader = csv.reader(f1)
		ranks = int(int(next(reader)[0].split(' ')[0]) / 200) - 1
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
			print(cp1)
			print(cp2)
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
			ave_OMSE[ranks] += omse
			ave_WMSE[ranks] += se2Mix(alpha, np.array(gt1), np.array(gt2), hatalpha, np.array(cp1), np.array(cp2), optimize=True, weighted=True)
			ave_SE1[ranks] += se1
			ave_time[ranks] += time
			countArray[ranks] += 1
		if omse > 2:
			array[ranks].append(alpha)
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
	plt.rcParams['xtick.labelsize'] = 20
	plt.rcParams['ytick.labelsize'] = 20
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
	for q in range(0,5):
		ave_OMSE[q] = ave_OMSE[q] / countArray[q]
		ave_WMSE[q] = ave_WMSE[q] / countArray[q]
		ave_time[q] = ave_time[q] / countArray[q]
		print("For", str((q + 1)*200), "rankings:")
		print("Average Optimum MSE:", ave_OMSE[q])
		print("Average Weighted Error:", ave_WMSE[q])
		print("Average Runtime:", ave_time[q], "\n\n")
	print(countArray)
	print(ave_OMSE)
	plt.plot(np.array(x), np.array(ave_OMSE), color='red', linewidth=4, label='Average Optimum MSE')
	plt.ylabel("Average Optimum MSE")
	plt.xlabel("# of Rankings")
	plt.show()
	print(ave_WMSE)
	plt.plot(np.array(x), np.array(ave_WMSE), color='blue', linewidth=4, label='Average Weighted MSE')
	plt.ylabel("Average Weighted MSE")
	plt.xlabel("# of Rankings")
	plt.show()
	print(ave_time)
	plt.plot(np.array(x), np.array(ave_time), color='green', linewidth=4, label='Average Runtime')
	plt.ylabel("Average Runtime")
	plt.xlabel("# of Rankings")
	plt.show()