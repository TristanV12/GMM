# Implementation of algorithm (3) from
# MM Algorithms for Generalized Bradley-Terry Models
# by David R. Hunter, 2004

import numpy as np
import aggregate
import plackettluce as pl
import time


class MMwBTAggregator(aggregate.RankAggregator):
    """
    Minorization-Maximization Rank Aggregation
    algorithm for the Plackett-Luce model
    """

    def aggregate(self, rankings, max_iters):
        """
        Description:
            Minorization-Maximization algorithm which returns an
            estimate of the ground-truth parameters, gamma for
            the given data.
        Parameters:
            rankings:  set of rankings to aggregate
            epsilon:   convergence condition value, set to None for iteration only
            max_iters: maximum number of iterations of MM algorithm
        """

        #weights = np.empty((1, self.m))[0]
        #for i in range(0, self.m-1):
        #    weights[i] = 1/(self.m - i - 1)

        # compute the matrix w, the numbers of pairwise wins:
        w = np.zeros((self.m, self.m))
        t0 = time.perf_counter()
        for ranking in rankings:
            for i1 in range(0, self.m-1):
                for i2 in range(i1+1, self.m):
                    w[int(ranking[i1]), int(ranking[i2])] += 1/(self.m-i1-1)#weights[i1]
        W = w.sum(axis=1)
        t1 = time.perf_counter()

        # gamma_t is the value of gamma at time = t
        # gamma_t1 is the value of gamma at time t = t+1 (the next iteration)
        # initial arbitrary value for gamma:
        gamma_t = np.ones(self.m) / self.m
        gamma_t1 = np.empty(self.m)
        gamma_rslt = np.empty((max_iters, self.m), float)

        for f in range(max_iters):

            for i in range(self.m):
                s = 0 # sum of updating function
                for j in range(self.m):
                    if j != i:
                        s += (w[j][i] + w[i][j]) / (gamma_t[i]+gamma_t[j])

                gamma_t1[i] = W[i] / s

            gamma_t1 /= np.sum(gamma_t1)
            gamma_rslt[f] = gamma_t1

            #if epsilon != None and np.all(np.absolute(gamma_t1 - gamma_t) < epsilon):
            #    alt_scores = {cand: gamma_t1[ind] for ind, cand in enumerate(self.alts)}
            #    self.create_rank_dicts(alt_scores)
            #    return gamma_t1 # convergence reached before max_iters

            gamma_t = gamma_t1 # update gamma_t for the next iteration

        t2 = time.perf_counter()
        alt_scores = {cand: gamma_t1[ind] for ind, cand in enumerate(self.alts)}
        self.create_rank_dicts(alt_scores)
        return gamma_rslt, (t1-t0), (t2-t1)


def main():
    """Driver function for the computation of the MM algorithm"""

    # test example below taken from GMMRA by Azari, Chen, Parkes, & Xia
    cand_set = [0, 1, 2]
    votes = [[0, 1, 2], [1, 2, 0], [0, 2, 1]]

    mmagg = MMPLAggregator(cand_set)
    gamma = mmagg.aggregate(votes, epsilon=1e-7, max_iters=30)
    print(mmagg.alts_to_ranks, mmagg.ranks_to_alts)
    #assert([mmagg.get_ranking(i) for i in cand_set] == [1,0,2])
    print(gamma)

if __name__ == "__main__":
    main()
