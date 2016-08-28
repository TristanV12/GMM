from generate import *
from EstimationNormalGMM import *
from likelihood_rum import *

if __name__ == '__main__':
	#Params = dict(m = 4, Mean = np.array([0.5,0.1,0.8,0]), SD = np.array([1,1,1,1]))
    mse = 0
    itr = 20
    n = 1000
    m = 4
    for iter in range(0, itr):
        print("Trial: ",iter)
        Params = GenerateRUMParameters(m, "normal")
        Data   = GenerateRUMData(Params,m,n,"normal")
		#print(Data)
        mu=Params["Mean"]
        mu=mu-mu.min()
        print("Ground truth: ", mu)
		#final = EstimationNormalGMM(mu, 4)
		#print(final["Parameters"])
        Data = Breaking(Data)
		#print(Data)
        final = EstimationNormalGMM(Data, m)
        muhat = []
        for i in range(0,m):
            muhat.append(final["Parameters"][i]["Mean"])
        print("Estimate: ", muhat)
        se = 0
        for i in range(0,m):
            se += (mu[i]-muhat[i])**2
        mse += se
        print("MSE:", se)
    mse = mse/itr
    print(mse)
