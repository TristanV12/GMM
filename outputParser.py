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
	array = [[]]*5
	x = [200, 400, 600, 800, 1000]
	files = glob.glob("output\\*.csv") + glob.glob("output2\\*.csv")

	trial = 0
	print(len(files))
	for file1 in files:
		trial += 1
		#print("Running trial", trial)
		f1 = open(file1, "r")
		reader = csv.reader(f1)
		ranks = int(int(next(reader)[0].split(' ')[0]) / 200) - 1
		next(reader)
		next(reader)
		next(reader)
		next(reader)
		next(reader)
		next(reader)
		arr = next(reader)[0].split(' ')
		#print("arr", arr
		if(len(arr) != 2):
			arr = next(reader)[0].split(' ')
		omse = float(arr[1])
		wmse = float(next(reader)[0].split(' ')[1])
		se1 = float(next(reader)[0].split(' ')[1])
		time = float(next(reader)[0].split(' ')[1])
		ave_OMSE[ranks] += omse / 200
		ave_WMSE[ranks] += wmse / 200
		ave_SE1[ranks] += se1 / 200
		ave_time[ranks] += time / 200
		array[ranks].append(omse)
		if omse > 1 and ranks == 0:
			print(omse)
		f1.close()
	array[0] = sorted(array[0])
	q0 = 0
	q1 = 0
	q2 = 0
	q3 = 0
	for q in range(0, len(array[0])):
		if array[0][q] > 2 and q0 == 0:
			q0 = q
			a = np.array(array[0][:q])
		elif array[0][q] > 6 and q1 == 0:
			q1 = q
			b = np.array(array[0][q0:q])
			print("b", b)
		elif array[0][q] > 11 and q2 == 0:
			q2 = q
			c = np.array(array[0][q1:q])
		elif array[0][q] > 17 and q3 == 0:
			q3 = q
			d = np.array(array[0][q2:q])
		elif array[0][q] > 25:
			e = np.array(array[0][q3:q])
			f = np.array(array[0][q:])
	# print(len(array[0]), len(array[1]), len(array[2]), len(array[3]), len(array[4]))
	# print(a)
	hfont = {'fontname':'Helvetica'}
	plt.hist(a, bins="auto")
	plt.rcParams['xtick.labelsize'] = 20
	plt.rcParams['ytick.labelsize'] = 20
	plt.ylabel("Frequency")
	plt.xlabel("Optimum MSE")
	plt.show()
	hfont = {'fontname':'Helvetica'}
	plt.hist(b, bins=[w / 10.0 for w in range(int(b.min()) - 1, int(b.max()) + 1)])
	plt.rcParams['xtick.labelsize'] = 20
	plt.rcParams['ytick.labelsize'] = 20
	plt.ylabel("Frequency")
	plt.xlabel("Optimum MSE")
	plt.show()
	hfont = {'fontname':'Helvetica'}
	plt.hist(c, bins=[w / 10.0 for w in range(int(c.min()) - 1, int(c.max()) + 1)])
	plt.rcParams['xtick.labelsize'] = 20
	plt.rcParams['ytick.labelsize'] = 20
	plt.ylabel("Frequency")
	plt.xlabel("Optimum MSE")
	plt.show()
	hfont = {'fontname':'Helvetica'}
	plt.hist(d, bins=[w / 10.0 for w in range(int(d.min()) - 1, int(d.max()) + 1)])
	plt.rcParams['xtick.labelsize'] = 20
	plt.rcParams['ytick.labelsize'] = 20
	plt.ylabel("Frequency")
	plt.xlabel("Optimum MSE")
	plt.show()
	hfont = {'fontname':'Helvetica'}
	plt.hist(e, bins=[w / 10.0 for w in range(int(e.min()) - 1, int(e.max()) + 1)])
	plt.rcParams['xtick.labelsize'] = 20
	plt.rcParams['ytick.labelsize'] = 20
	plt.ylabel("Frequency")
	plt.xlabel("Optimum MSE")
	plt.show()
	hfont = {'fontname':'Helvetica'}
	plt.hist(f, bins=[w / 10.0 for w in range(int(f.min()) - 1, int(f.max()) + 1)])
	plt.rcParams['xtick.labelsize'] = 20
	plt.rcParams['ytick.labelsize'] = 20
	plt.ylabel("Frequency")
	plt.xlabel("Optimum MSE")
	plt.show()
	# for q in range(0,5):
	# 	print("For", str((q + 1)*200), "rankings:")
	# 	print("Average Optimum MSE:", ave_OMSE[q])
	# 	print("Average Weighted Error:", ave_WMSE[q])
	# 	print("Average Runtime:", ave_time[q], "\n\n")
	# print(ave_OMSE)
	# plt.plot(np.array(x), np.array(ave_OMSE), color='red', label='Average Optimum MSE')
	# plt.ylabel("Average Optimum MSE")
	# plt.xlabel("# of Rankings")
	# plt.show()
	# print(ave_WMSE)
	# plt.plot(np.array(x), np.array(ave_WMSE), color='blue', label='Average Weighted MSE')
	# plt.ylabel("Average Weighted MSE")
	# plt.xlabel("# of Rankings")
	# plt.show()
	# print(ave_SE1)
	# plt.plot(x, ave_SE1, color='green', label='Average SE1')
	# plt.ylabel("Average SE1")
	# plt.xlabel("# of Rankings")
	# plt.show()
	# print(ave_time)
	# plt.plot(np.array(x), np.array(ave_time), color='orange', label='Average Runtime')
	# plt.ylabel("Average Runtime")
	# plt.xlabel("# of Rankings")
	# plt.show()