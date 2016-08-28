from generate import *
from EstimationNormalGMM import *
from likelihood_rum import *
from EGMM import *

if __name__ == '__main__':
	Params = dict(m=4, Mean=np.array([0.8902497,0.7672302,0.5470003,0.1942752]), SD=np.array([0.7698108,0.4167230,0.2128372,0.8178268]))
	Data   = np.array([[3,2,1,4]])
	print(Params["Mean"])
	print(likelihoodRUM(Data, Params))