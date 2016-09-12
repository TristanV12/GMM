'''
	Title: E-GMM Algorithm
	Authors: 	Tristan Villamil
				Lirong Xia
				Zhibing Zhao

	Description: The algorithm
'''
import sys
import time
import pickle
import numpy as np				  # used for matrices and vectors
import time						  # used for getting runtime
from likelihood_rum import *	  # used for getting likelihood
from EstimationNormalGMM import * # used for GMM
from generate import *			  # used for breaking
from GaussianRUMGMM import *	  # used for GMM
from scipy.stats import rankdata
from scipy.optimize import minimize

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

#This function was modified by Zhibing to have reversed random initial values for k = 2.
def getTheta(m, k):
	#variables
	SD = np.array([1] * m) # Standard deviation
	#inner = list(range(m))	# temporary array
	#counter = int(m / k)	# used for greater randomization

	#for itr in range(0, len(inner)):
	#	inner[itr] *= .1

	#arr = [dict(Mean=np.array(inner), SD=SD)]  # final array of RUMs
	arr = [dict(Mean=np.array([0, 2, 4, 6, 8, 10]), SD=SD)]
	for i in range(1, k):
	# generate k RUMs - 1,k because the first is initialized
	#for i in range(1, k):
	#	if i % 2 == 1:
	#		inner.reverse()
	#	else:
	#		inner = list(range(counter * i, k)) + list(range(0, counter * i))
		arr.append(dict(Mean=arr[0]["Mean"][::-1], SD=SD))
	#	inner = list(inner) # copy list, don't alias
	return arr

"""
	To calculate the squared error between ground truth and estimated results for k = 2
	When optimize = True, the shifts are selected s.t. the squared error is minimized
	When optimize = False and normalize = True, the means are normalized s.t. the sum of exponentials are 1
	When optimize = False and normalize = False, the means are normalized s.t. the minimum mean is 0
"""
def se2Mix(alpha, gt1, gt2, hatalpha, cp1, cp2, optimize = True, weighted = True, normalize = False):
	if optimize:
		m = len(gt1)
		d11 = (np.sum(gt1) - np.sum(cp1))/m
		d12 = (np.sum(gt2) - np.sum(cp2))/m
		d21 = (np.sum(gt1) - np.sum(cp2))/m
		d22 = (np.sum(gt2) - np.sum(cp1))/m
		if weighted:
			se1 = hatalpha * np.sum((gt1 - cp1 - d11) ** 2) + (1 - hatalpha) * np.sum((gt2 - cp2 - d12) ** 2) + (alpha - hatalpha) ** 2
			se2 = (1 - hatalpha) * np.sum((gt1 - cp2 - d21) ** 2) + hatalpha * np.sum((gt2 - cp1 - d22) ** 2) + (alpha + hatalpha - 1) ** 2
		else:
			se1 = np.sum((gt1 - cp1 - d11) ** 2) + np.sum((gt2 - cp2 - d12) ** 2) + (alpha - hatalpha) ** 2
			se2 = np.sum((gt1 - cp2 - d21) ** 2) + np.sum((gt2 - cp1 - d22) ** 2) + (alpha + hatalpha - 1) ** 2
		if se1 <= se2:
			return se1, cp1+d11, cp2+d12
		else:
			return se2, cp1+d22, cp2+d21
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

"""
	Generate data for Gaussian distributed random utility models.
	Simplied from GenerateRUMData
"""

def GenGauRUM(means, n):
	m = len(means)
	A = rankdata(-np.random.normal(size = m, loc = means, scale = 1))-1
	for i in range(0, n - 1):
		A = np.vstack([A, (-np.random.normal(size = m, loc = means, scale = 1)).ravel().argsort()])
	return A.astype(int)

"""
	Convert a ranking to a string so that it can be used as keys to dictionaries.
"""

def rank2str(ranking):
	s = ""
	for alt in ranking:
		s += str(alt)
	return s

"""
	Given a dataset, this function returns a dictionary with rankings as keys and frequencies of each ranking as values
"""

def Dictionarize(rankings):
	rankcnt = {}
	for ranking in rankings:
		key = rank2str(ranking)
		if key in rankcnt:
			rankcnt[key] += 1
		else:
			rankcnt[key] = 1
	return rankcnt

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
def EGMM(alphastar, GT1, GT2, data, m, k=2, itr=5):
	#get start time
	t0 = time.time()

	#initialize variables
	alphas = np.array([[np.random.rand()], [np.random.rand()]])
	alphas = alphas/np.sum(alphas)
	print("InitAlphas: ", alphas)
	n = len(data)
	thetas = getTheta(m, k) # RUMs
	#print("THETAS: ", thetas)
	likelihoods = []		# likelihoods
	ll = []					# temporary list of likelihoods
	breaking = np.zeros((k, m, m), float)

	for i in range(0, itr):
		output = np.zeros((k, m), float)
		print("Itr: ", i)
		#print("\b"*len(str(i-1)) + str(i), end='')
		#print("E Step: ")
		for theta in thetas:
			for vote in data:
				ll.append(likelihoodRUM(vote, theta))
			likelihoods.append(ll)
			ll = []
		weights = alphas*likelihoods
		weights = np.divide(weights, np.sum(weights, axis = 0))
		#print("WEIGHTS:", weights)
		#print("Calculating breakings...")
		breaking = dataBreaking(data, m, k, weights)
		#print("BREAKING: ", breaking)
		#print("M-step:")
		alphas = (np.sum(weights, axis=1)).reshape(k, 1)
		for r in range(0, k):
			output[r] = GMMGaussianRUM(breaking[r], m, alphas[r])
			thetas[r] = dict(Mean=output[r], SD=np.array([1] * m))
		alphas = alphas/n
		likelihoods = []
		se1 = np.sum((GT1-thetas[0]["Mean"]) ** 2) + np.sum((GT2-thetas[1]["Mean"]) ** 2) + (alphastar - alphas[0]) ** 2
		se2 = np.sum((GT1-thetas[1]["Mean"]) ** 2) + np.sum((GT2-thetas[0]["Mean"]) ** 2) + (alphastar - alphas[1]) ** 2
		print("ERROR: ", min(se1, se2))
	tf = time.time()

	return alphas, thetas

"""
	Another version of EGMM algorithm. This version uses sampling method to estimate the probability of a ranking given a ground truth.
"""

def E_GMM(alphastar, GT1, GT2, data, m, k = 2, itr = 20, n0 = 10000, ns = 2000):
	#get start time
	t0 = time.time()
	print("Initializing...")
	#initialize variables
	alphas = np.zeros((k, 1), float)
	alphasp = np.zeros((k, 1), float)
	finalItr = itr
	#Random initialization of alphas.
	for r in range(0, k):
		#alphas[r][0] = 1/k
		alphas[r] = np.random.rand()
	alphas = alphas/np.sum(alphas)
	# print("InitAlphas: ", alphas)

	n = len(data)
	DataDict = Dictionarize(data)#A dictionay of dataset

	#print("DataDict: ", DataDict)
	weights = [] #To store dictionaries of rankings in the dataset given different components.
	for r in range(0, k):
		weights.append(DataDict.copy()) # The length of each dictionary is the same as the dataset dictionary
	#print("Weights: ", weights)
	thetas = getTheta(m, k) # RUMs
	#print("Thetas: ", thetas)
	freqdicts = [] # To store samples from each component
	breaking = np.zeros((k, m, m), float) # Breakings for each component
	output = np.zeros((k, m), float) # Estimated means for each component

	for i in range(0, itr):
		print("EM Itr: ", i)
		ini = np.random.uniform(0,5,(k,m))
		#print(ini)
		n1 = np.zeros((1, k), float)[0]
		#print("E Step: ")
		for r in range(0, k):
			rankings = GenGauRUM(output[r], n0)
			#freqdicts.append(Dictionarize(rankings))
			tempdict = Dictionarize(rankings)
			for key, value in tempdict.items():
				tempdict[key] /= n0
			freqdicts.append(tempdict)
		ss = 0
		for vote, freq in DataDict.items():
			for r in range(0, k):
				if vote in freqdicts[r]:
					#print("Prob for cp ", r, " for vote ", vote, "is", freqdicts[r][vote])
					weights[r][vote] = alphas[r][0] * freqdicts[r][vote]
					ss += weights[r][vote]
				else:
					weights[r][vote] = 0 #ignore this vote if it is not in the samples
			if ss != 0:
				#print(ss)
				for r in range(0, k):
					weights[r][vote] /= ss
					#print("Weights for cp ", r, " for vote ", vote, " is ", weights[r][vote], "Freq: ", freq)
					weights[r][vote] *= freq
			else:
				for r in range(0, k):
					weights[r][vote] = freq * alphas[r][0]
			ss = 0
		breaking = freqBreaking(weights, m, k)
		#print("BREAKING: ", breaking)
		#print("M-step:")
		total = 0
		for r in range(0, k):
			for key, value in weights[r].items():
				n1[r] += value
			total += n1[r]
			#print("n1", r, ":", n1[r])
		for r in range(0, k):
			alphasp[r][0] = n1[r]/total
			breaking[r] = breaking[r] * ns / n1[r]
			#output[r] = GMMGaussianRUM(output[r], breaking[r], m, n1[r])
			#output[r] = GMMGaussianRUM(output[r], breaking[r], m)
			rslt = minimize(GauRUMobj, x0=ini[r], args=(breaking[r]), bounds = ((0, 5),(0, 5),(0, 5),(0, 5), (0, 5), (0, 5)))
			output[r] = rslt.x
			#print("Output", r, output[r])
			output[r] -= output.min()#np.log(np.sum(np.exp(output[r])))

		freqdicts = []
		se, cp0, cp1 = se2Mix(alphastar, GT1, GT2, alphas[0][0], output[0], output[1], optimize = True, weighted = False)
		wse, cp0, cp1 = se2Mix(alphastar, GT1, GT2, alphas[0][0], output[0], output[1], optimize = True, weighted = True)
		# print("Optimal ERROR: ", se)
		# print("Weighted ERROR: ", wse)

		se1, cp10, cp11 = se2Mix(alphas[0][0], thetas[0]["Mean"], thetas[1]["Mean"], alphasp[0][0], output[0], output[1], optimize = True, weighted = False)
		# print("SE1: ", se1)
		#print(alphas[0, 0], alphasp[0, 0], thetas[0]["Mean"], thetas[1]["Mean"], output)
		alphas = alphasp
		for r in range(0, k):
			thetas[r] = dict(Mean=output[r].copy(), SD=np.array([1] * m))
		#print("EstMeans: for ", r, "is", output[r])

		# if se1 <= 1e-3 and i >= 5:
		# 	finalItr = i
		# 	break

	tf = time.time()

	return alphas, thetas, cp0, cp1, tf - t0, se, wse, se1, finalItr
