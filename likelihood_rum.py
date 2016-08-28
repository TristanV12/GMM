import numpy as np
from scipy.stats import norm
import time
import math

# ' Likelihood for general Random Utility Models
# '
# ' @param Data ranking data
# ' @param parameter Mean of Exponential Distribution
# ' @param range range
# ' @param res res
# ' @param race TRUE if data is sub partial, FALSE (default) if not
# ' @return log likelihood
# ' @export
# ' @examples
# ' data(Data.Test)
# ' parameter = Generate.RUM.Parameters(5, "normal")
# ' Likelihood.RUM(Data.Test,parameter, "norm")
'''
    Title: likelihoodRUM
    Authors:    Tristan Villamil
                Lirong Xia
                Zhibing Zhao

    Description: The algorithm

    Input:
        Data      - the ranking
                        numpy array of rankings, see generate.py GenerateRUMData for data generation.
        parameter - mean of exponental distribution
                        numpy array of means, see generate.py GenerateRUMParameters for generation
        range_var - range
                        float
        res       - res
                        float

    Output:
        likelihood that Data belongs to parameter
            float
'''
def likelihoodRUM(Data, parameter, range_var = None, res = None):
    t0 = time.time()
    if range_var == None:
        range_var = parameter["Mean"].max() + 3 * parameter["SD"].max()
    if res == None:
        res = range_var / 10000

    rank = Data
    S = range_var / res
    x = []
    for i in range(-int(S), int(S) + 1):
        x.append(i * res)

    n = rank.shape[0]
    m = 4

    ll = 0
    CDF = np.ones((1,len(x)), int)
    for j in range(m - 1, -1,-1):
        PDF = norm.pdf(x, loc=parameter["Mean"][rank[j] - 1],scale=parameter["SD"][rank[j] - 1])*CDF
        CDF = res * PDF.cumsum()
    ll = CDF[len(x) - 1]
    tf = time.time()
    #print("likelihood time:", tf - t0)
    return ll