# Implementation of algorithm (3) from
# MM Algorithms for Generalized Bradley-Terry Models
# by David R. Hunter, 2004

import numpy as np
import aggregate
import plackettluce as pl
import util
import time


class MM_PL(aggregate.RankAggregator):
    """
    Minorization-Maximization Rank Aggregation
    algorithm for the Plackett-Luce model
    """

    def ll(self, rankings, gamma):
        ll = 0
        for ranking in rankings:
            for i1 in range(self.m - 1):
                localsum = 0
                ll += np.log(gamma[int(ranking[i1])])
                for i2 in range(i1, self.m):
                    localsum += gamma[ranking[i2]]
                ll -= np.log(localsum)
        return ll

    def aggregate(self, rankings, max_iters):
        """
        Description:
            Minorization-Maximization algorithm which returns an
            estimate of the ground-truth parameters, gamma for
            the given data.
        Parameters:
            rankings:  set of rankings to aggregate
            max_iters: maximum number of iterations of MM algorithm
        """

        gamma0 = np.random.rand(self.m)
        n = len(rankings)

        gamma_t = gamma0 / np.sum(gamma0)
        gamma_t1 = np.empty(self.m)
        gamma_rslt = np.empty((max_iters, self.m+1), float)

        w = n*np.ones((1, self.m), int)[0]

        t0 = time.perf_counter()
        for ranking in rankings:
            w[int(ranking[-1])] -= 1

        for f in range(max_iters):
            #print("Iter: ", f)

            for i1 in range(self.m):
                denom = 0
                for ranking in rankings:
                    #print("Ranking: ", ranking)
                    for i2 in range(self.m-1):
                        localsum = 0
                        for i3 in range(i2, self.m):
                            localsum += gamma_t[int(ranking[i3])]
                        #print("Gamma: ", gamma_t)
                        #print("LocalSum: ", localsum)
                        denom += 1/localsum
                        if int(ranking[i2]) == i1:
                            #print("Find the alternative!")
                            break
                gamma_t1[i1] = w[i1] / denom
            gamma_t1 = gamma_t1/np.sum(gamma_t1)
            gamma_t = gamma_t1
            t1 = time.perf_counter() - t0
            gamma_rslt[f, 0:self.m] = gamma_t1
            gamma_rslt[f, self.m] = t1
            #print("Likelihood: ", self.ll(rankings, gamma_t1))
        return gamma_rslt


def main():
    """Driver function for the computation of the MM algorithm"""

    # test example below taken from GMMRA by Azari, Chen, Parkes, & Xia
    cand_set = [0, 1, 2]
    votes = [[0, 1, 2], [1, 2, 0]]
    #print("Data: ", votes)

    mmagg = MM_PL(cand_set)
    gamma = mmagg.aggregate(votes, max_iters=30)

    print(gamma)

if __name__ == "__main__":
    main()
