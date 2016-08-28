from generate import *
import csv
import sys
import glob
import os

def generateTrial(trial, rankings):
	filename = '/data/M6K2Trial' + str(trial) + '.csv'
	os.makedirs(os.path.dirname(filename), exist_ok=True)
	file1 = open(filename, 'a')
	parameters = GenerateRUMParameters(6, "normal")
	arr = GenerateRUMData(parameters, 6, rankings, "normal")
	spamwriter = csv.writer(file1, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
	spamwriter.writerow(parameters["Mean"])
	for a in arr:
		spamwriter.writerow(a)

if __name__ == '__main__':
	trials = 1000
	rankings = 10000

	for t in range(0, trials):
		print("Generating trial ", t)
		generateTrial(t, rankings)