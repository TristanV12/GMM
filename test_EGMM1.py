from generate import *
from EstimationNormalGMM import *
from likelihood_rum import *
from EGMM import *
from GaussianRUMGMM import *

if __name__ == '__main__':
	Params = GenerateRUMParameters(6, "normal")
	Data   = GenerateRUMData(Params,6,100,"normal")
	print("ground truth", Params)
	print("\n\nGround truth order", Params["order"])
	#print(likelihoodRUM(Data,Params, "norm"))
	DataPairs = Breaking(Data)
	#print("\n\nData", Data)
	final = EGMM(Data, DataPairs, 6)
	print(final)
	mu = Params["Mean"]
	mu = mu - mu.min()
	muhat = final[0]["Mean"]
	muhat = muhat - muhat.min()
	mse = 0
	for itr in range(0,len(muhat)):
		mse += (muhat[itr] - mu[itr]) ** 2
	print("MSE:", mse)