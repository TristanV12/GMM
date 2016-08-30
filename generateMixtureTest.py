from generate import *
import csv
import sys
import glob
import os
import random

def generateTrial(trial, rankings):
	file1 = open('M6K2Trial' + str(trial) + '.csv', 'a')
	parameters = GenerateRUMParameters(6, "normal")
	arr = GenerateRUMData(parameters, 6, rankings, "normal")
	writer = csv.writer(file1, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
	writer.writerow(parameters["Mean"])
	for a in arr:
		writer.writerow(a)

if __name__ == '__main__':
	trials = 1000
	rankings = 10000

	# for t in range(0, trials):
	# 	print("Generating trial ", t)
	# 	generateTrial(t, rankings)
	files = glob.glob("*.csv")
	count = 0
	while count < len(files):
		alpha = random.random()
		f1 = open(files[count])
		if count + 1 != rankings:
			f2 = open(files[count + 1])
		else:
			f2 = open(files[0])
		file1 = open('M6K2Trial' + str(int(count / 2)) + 'Mixture.csv', 'a')
		writer = csv.writer(file1, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		reader1 = csv.reader(f1)
		reader2 = csv.reader(f2)
		mean1 = next(reader1)[0].split(' ')
		mean2 = next(reader2)[0].split(' ')
		writer.writerow(mean1)
		writer.writerow(mean2)
		writer.writerow([alpha])
		for itr in range(0, rankings):
			a = random.random()
			if a < alpha:
				next(reader1)
				writer.writerow(next(reader1)[0].split(' '))
			else:
				next(reader2)
				writer.writerow(next(reader2)[0].split(' '))
		count += 1