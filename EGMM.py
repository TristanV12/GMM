'''
	Title: E-GMM Algorithm
	Authors: 	Tristan Villamil
				Lirong Xia
				Zhibing Zhao

	Description: The algorithm
'''
import numpy as np				  # used for matrices and vectors
import time						  # used for getting runtime
from likelihood_rum import *	  # used for getting likelihood
from EstimationNormalGMM import * # used for GMM
from generate import *			  # used for breaking
from GaussianRUMGMM import *	  # used for GMM

def getNewAlphas(D):
	arr = []
	count = 0
	for C in D:
		arr.append(0)
		for row in C:
			for col in row:
				if not math.isnan(col):
					arr[count] += col
		count += 1
	total = 0
	for a in arr:
		total += a
	for i in range(0, len(arr)):
		arr[i] = arr[i] / total
	return arr



def normalizeD(D):
	total = np.zeros((len(D[0]), len(D[0])), float)
	for i in range(0,len(D)):
		total += D[i]
	for j in range(0, len(D)):
		D[j] = D[j] / total
	return D

'''

'''
def generateC(Data, alphas, ll, m, k):
	# C is the transition matrix, where C[i, j] denotes the number of times that
	C = np.zeros((m, m), float)
	D = []

	for i in range(0, k):
		for j in range(0, len(Data)):
			C[Data[j][0], Data[j][1]] += alphas[i] * ll[i][Data[j][3]]
		D.append(C)
		C = np.zeros((m, m), float)
	return normalizeD(D)

'''
	Title: getTheta Function

	Function: Given m and k, return an array of RUMs

	Input:
		m	 - the number of alternatives (candidates)
				integer
		k	 - the number of components
				integer
				must be greater than or equal to 1
				default is 2

	Output: Array of possible RUMs
'''
def getTheta(m, k):
	#variables
	SD = np.array([1] * m) # Standard deviation
	arr = [dict(Mean=np.array(list(range(m))), SD=SD)]  # final array of RUMs
	inner = list(range(m))	# temporary array
	counter = int(m / k)	# used for greater randomization 

	# generate k RUMs - 1,k because the first is initialized
	for i in range(1, k):
		if i % 2 == 1:
			inner.reverse()
		else:
			inner = list(range(counter * i, k)) + list(range(0, counter * i))

		arr.append(dict(Mean=np.array(inner), SD=SD))
		inner = list(inner) # copy list, don't alias
	return arr

'''
	Title: EGMM Function

	Function: Algorithm

	Input:
		data - the rankings
				numpy array of rankings, see generate GenerateRUMData for data generation.
		n 	 - the number of agents (rankings)
				integer
		m	 - the number of alternatives (candidates)
				integer
		k	 - the number of components
				integer
				must be greater than or equal to 1
				default is 2
		itr	 - the number of iterations
				integer
				default is 10

	Output:

'''
def EGMM(data, datapairs, m, k=2, itr=20):
	#get start time
	t0 = time.time()

	#initialize variables
	alpha = [1/k] * k
	thetas = getTheta(m, k) # RUMs
	likelihoods = []		# likelihoods
	ll = []					# temporary list of likelihoods

	for i in range(0, itr):
		for theta in thetas:
			for vote in data:
				ll.append(likelihoodRUM(vote, theta))
			likelihoods.append(ll)
			ll = []

		D = generateC(datapairs, alpha, likelihoods, m, k)

		for x in range(0, len(thetas)):
			output = GMMGaussianRUM(data, D[x], m, len(likelihoods[0]))
			thetas[x] = dict(Mean=output[0], SD=np.array([1] * m))
		print("\nthetas", thetas)
		alpha = getNewAlphas(D)
		print("alpha", alpha)
	tf = time.time()
	print("Total runtime: ", tf - t0)

	return thetas