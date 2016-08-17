from generate import *
from EstimationNormalGMM import *
from likelihood_rum import *

if __name__ == '__main__':
	Params = GenerateRUMParameters(4, "normal")
	Data   = GenerateRUMData(Params,4,2,"normal")
	Params = dict(m = 4, Mean = np.array([0.26526090,0.04041244,0.22239384,0.34170505]), SD = np.array([0.30001765,0.48085526,0.06827066,0.64942853]))
	Data = np.array([[2,4,3,1],[4,1,3,2]])
	print(Data)
	print(Params)
	Breaking(Data)
	print(likelihoodRUM(Data,Params, "norm"))