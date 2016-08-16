from generate import *
from EstimationNormalGMM import *
from likelihood_rum import *

if __name__ == '__main__':
	Params = GenerateRUMParameters(4, "normal")
	Data   = GenerateRUMData(Params,4,3,"normal")
	print(Data)
	print(likelihoodRUM(Data,Params, "norm"))