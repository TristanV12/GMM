from generate import *
import csv
import sys
import glob
import os
import random

def generateTrial(trial, rankings):
	file1 = open('SpecialData\\M6K2RUM' + str(trial) + '.csv', 'a')
	parameters = GenerateRUMParameters(6, "normal")
	arr = GenerateRUMData(parameters, 6, rankings, "normal")
	spamwriter = csv.writer(file1, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
	spamwriter.writerow(parameters["Mean"])
	for a in arr:
		spamwriter.writerow(a)

if __name__ == '__main__':
	trials = 2
	rankings = 10000

	for t in range(1, trials + 1):
		print("Generating trial ", t)
		generateTrial(t, rankings)

	f1 = open('SpecialData\\M6K2RUM1.csv')
	f2 = open('SpecialData\\M6K2RUM2.csv')
	alpha = random.random()
	output1 = open('M6K2Mixture.csv', 'a')
	output2 = open('M6K2MixtureAnnotated.csv', 'a')
	writer1 = csv.writer(output1, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
	writer2 = csv.writer(output2, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
	reader1 = csv.reader(f1)
	reader2 = csv.reader(f2)
	mean1 = next(reader1)[0].split(' ')
	mean2 = next(reader2)[0].split(' ')
	writer1.writerow(mean1)
	writer1.writerow(mean2)
	writer1.writerow([alpha])
	mean1.append("RUM1")
	mean2.append("RUM2")
	writer2.writerow(mean1)
	writer2.writerow(mean2)
	writer2.writerow([alpha, "alpha"])
	counter = 0
	print(int(rankings / 2))
	for i in range(0, int(rankings / 2)):
		print(i)
		counter += 1
		a = random.random()
		if a < alpha:
			next(reader1)
			vote = next(reader1)[0].split(' ')
			writer1.writerow(vote)
			vote.append("RUM1")
			writer2.writerow(vote)
		else:
			next(reader2)
			vote = next(reader2)[0].split(' ')
			writer1.writerow(vote)
			vote.append("RUM2")
			writer2.writerow(vote)