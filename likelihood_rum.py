import numpy as np
from scipy.stats import norm
import time
import math

# ' Likelihood for general Random Utility Models
# '
# ' @param Data ranking data
# ' @param parameter Mean of Exponential Distribution
# ' @param dist exp or norm
# ' @param range range
# ' @param res res
# ' @param race TRUE if data is sub partial, FALSE (default) if not
# ' @return log likelihood
# ' @export
# ' @examples
# ' data(Data.Test)
# ' parameter = Generate.RUM.Parameters(5, "normal")
# ' Likelihood.RUM(Data.Test,parameter, "norm")
def likelihoodRUM(Data, parameter, dist = "exp", range_var = None, res = None, race = False):
    if range_var == None:
        range_var = parameter["Mean"].max() + 3 * parameter["SD"].max()
    if res == None:
        res = range_var / 10000

    if not(dist=="fexp" or dist=="dexp" or dist=="exp" or dist=="norm" or dist=="norm.fixedvariance"):
        raise ValueError("Distribution name \"", dist, "\" not recognized")

    rank = Data
    S = range_var / res
    if dist == "fexp" or dist == "dexp" or dist == "exp":
        x = []
        for i in range(0, int(S) + 1):
            x.append(i * res)
    else:
        x = []
        for i in range(-int(S), int(S) + 1):
            x.append(i * res)

    n = rank.shape[0]
    m = rank.shape[1]

    ll = 0
    if dist == "norm" or dist == "norm.fixedvariance":
        for i in range(0, n):
            if (Data[i,] == 0).sum() > 0:
                mj = (rank[i,] == 0).min() - 1
                CDF = ones((1,len(x)), Float)
                if not race:
                    for jt in setdiff1d(range(1, m + 1), rank[i,1:mj]):
                        CDF = norm.pdf(x, mean=parameter["Mean"][jt],sd=parameter["SD"][jt]).cumsum() * CDF * res
                for j in range(mj, 2):
                    PDF = norm.pdf(x, mean=parameter["Mean"][rank[i,j]],sd=parameter["SD"][rank[i,j]])
                    CDF = res * PDF.cumsum()
                ll = math.log(CDF[len(x)]) + ll
            else:
                CDF = np.ones((1,len(x)), int)
                for j in range(m, 1):
                    PDF = norm.pdf(x, loc=parameter["Mean"][int(rank[i]) - 1],scale=parameter["SD"][int(rank[i]) - 1])
                    CDF = res * PDF.cumsum()
                print(CDF)
                ll = math.log(CDF[len(x) - 1]) + ll
    return ll

#     if dist == "exp":
#         ll = 0
#         for i in range(1, n + 1):
#             if (Data[i,] == 0).sum() > 0:
#                 mj = rank[i,].index(0)
#             CDF = ones((1,len(x)), Float)
#             if !race:
#                for jt in list(set(range(1,m + 1)) - set(rank[i,1:mj])):
#                   CDF = res*cumsum(dexp(x, rate=1/parameter["Mean"][jt]))*CDF



  
    
#   if(dist=="exp"){
#    ll <- 0
#    for(i in 1:n){
#      if(sum(Data[i,]==0)>0){
#       mj <- min(which(rank[i,]==0))-1
#       CDF <- matrix(1,1,length(x))
#       if(!race){
#         for(jt in setdiff(1:m,rank[i,1:mj])){
#          CDF <- res*cumsum(dexp(x,rate=parameter$Mean[jt]^-1))*CDF
#         }
#       }
        
#       for(j in mj:1){
#         PDF <- dexp(x,rate=parameter$Mean[rank[i,j]]^-1)*CDF
#         CDF <- res*cumsum(PDF)
#         CDF <- CDF[length(CDF)]-CDF
#       }
#       ll <- log(CDF[1])+ll
#      }
      
#      if(!(sum(Data[i,]==0)>0)){
#       CDF <- matrix(1,1,length(x))
#       for(j in m:1){
#         PDF <- dexp(x,rate=parameter$Mean[rank[i,j]]^-1)*CDF
#         CDF <- res*cumsum(PDF)
#         CDF <- CDF[length(CDF)]-CDF
#       }
#       ll <- log(CDF[1])+ll
#      }
#    }
#   }
  
#   if(dist=="fexp"){
#    ll <- 0
#    for(i in 1:n){
#      if(sum(Data[i,]==0)>0)
#      {
#       mj <- min(which(rank[i,]==0))-1
#       CDF <- matrix(1,1,length(x))
#       for(j in 1:mj){
#         PDF <- dexp(x,rate=parameter$Mean[rank[i,j]]^-1)*CDF
#         CDF <- res*cumsum(PDF)
#         CDF <- CDF[length(CDF)]-CDF
#       }
        
#       if(!race){
#         for(jt in setdiff(1:m,rank[i,1:mj]) ){
#          PDF <- res*cumsum(dexp(x,rate=parameter$Mean[jt]^-1))*CDF
#         }
#         CDF <- res*cumsum(PDF)
#         CDF <- CDF[length(CDF)]-CDF
#       }
#       ll <- log(CDF[1])+ll
#      }
      
#      if(!(sum(Data[i,]==0)>0)){
#       CDF <- matrix(1,1,length(x))
#       for(j in m:1){
#         PDF <- dexp(x,rate=parameter$Mean[rank[i,j]]^-1)*CDF
#         CDF <- res*cumsum(PDF)
#         CDF <- CDF[length(CDF)]-CDF
#       }
#       ll <- log(CDF[1])+ll
#      }
#    }
#   }
  
#   if(dist=="dexp"){
#    x <- (-S:S)*res
#    ll <- 0
#    for(i in 1:n){
#      CDF <- matrix(1,1,length(x))
#      for(j in m:1){
#       PDF <- pdfgumbel(x,mu=parameter$Mean[rank[i,j]])*CDF
#       CDF <- res*cumsum(PDF)
#      }
#      ll <- log(CDF[length(x)])+ll
#    }
#   }
  
#   ll
# }