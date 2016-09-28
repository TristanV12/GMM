from generate import *
from likelihood_rum import *
from EGMM import *
from GaussianRUMGMM import *
from PL_MLE import *
import csv
import sys
import glob
import os

if __name__ == '__main__':
	Data = np.array([[7,10,9,2,1,5,3,4,8,6],[9,10,8,5,2,7,4,6,1,3],[10,9,1,7,5,8,2,4,6,3],[2,5,9,1,10,4,3,7,8,6],[3,9,8,10,2,1,4,5,6,7]])
	print(Estimation_PL_MLE(Data))