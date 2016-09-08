from generate import *
import csv
import sys
import glob
import os
import random

if __name__ == '__main__':
	trials = 1000
	rankings = 10000
	p2 = GenerateRUMParameters(6, "normal")

	for count in range(0, trials):
		print(count)
		alpha = random.random()
		file1 = open('M6R10000Trial' + str(int(count)) + 'Mixture.csv', 'a')
		writer = csv.writer(file1, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		p1 = p2
		p2 = GenerateRUMParameters(6, "normal")
		mean1 = p1["Mean"]
		mean2 = p2["Mean"]
		writer.writerow(mean1)
		writer.writerow(mean2)
		writer.writerow([alpha])

		mean1Rankings = GenerateRUMData(p1, 6, rankings, "normal")
		mean2Rankings = GenerateRUMData(p2, 6, rankings, "normal")
		itr1 = 0
		itr2 = 0

		for itr in range(0, rankings):
			a = random.random()
			if a < alpha:
				writer.writerow(mean1Rankings[itr1])
				itr1 += 1
			else:
				writer.writerow(mean2Rankings[itr2])
				itr2 += 1
		file1.close()