from generate import *
import csv
import sys
import glob
import os
import random

if __name__ == '__main__':
	trials = 1000
	rankings = 10000

	files = glob.glob("*.csv")
	count = 0
	print(len(files))
	while count < len(files):
		print(count)
		alpha = random.random()
		f1 = open(files[count])
		if count + 1 != len(files):
			f2 = open(files[count + 1])
		else:
			f2 = open(files[0])
		file1 = open('M6K2Trial' + str(int(count)) + 'Mixture.csv', 'a')
		writer = csv.writer(file1, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		reader1 = csv.reader(f1)
		reader2 = csv.reader(f2)
		mean1 = next(reader1)[0].split(' ')
		mean2 = next(reader2)[0].split(' ')
		writer.writerow(mean1)
		writer.writerow(mean2)
		writer.writerow([alpha])
		counter = 0
		for itr in range(0, 10000):
			print(counter)
			counter += 1
			a = random.random()
			if a < alpha:
				next(reader1)
				writer.writerow(next(reader1)[0].split(' '))
			else:
				next(reader2)
				writer.writerow(next(reader2)[0].split(' '))
		count += 1 