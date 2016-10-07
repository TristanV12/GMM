# Implementation of the algorithm from
# Generalized Method-of-Moments for Rank
# Aggregation by Azari, Chen, Parkes, & Xia

import numpy as np
import aggregate
import plackettluce as pl
import time


class GMMPLAggregator(aggregate.RankAggregator):
    """
    Generalized Method-of-Moments for Rank Aggregation
    algorithm for the Plackett-Luce model
    """

    def aggregate(self, rankings, k = None):
        """
        Description:
            Takes in a set of rankings and computes the
            Plackett-Luce model aggregate ranking.
        Parameters:
            rankings: set of rankings to aggregate
            k:        number to be used for top-k or full breakings
            Specifically, for top-1 breaking, let k = 1
        """

        if k == None or k >= self.m: #Full breaking
            k = self.m - 1

        P = np.zeros((self.m, self.m))
        t0 = time.perf_counter()
        for ranking in rankings:
            for i1 in range(0, k):
                for i2 in range(i1+1, self.m):
                    P[int(ranking[i1]), int(ranking[i2])] += 1
        for ind in range(0, self.m):
            P[ind][ind] = -(np.sum(P.T[ind]))
        P /= len(rankings)
        t1 = time.perf_counter()

        U, S, V = np.linalg.svd(P)
        gamma = np.abs(V[-1])
        gamma /= np.sum(gamma)
        t2 = time.perf_counter()
        #assert(all(np.dot(P, gamma) < epsilon))
        alt_scores = {cand: gamma[ind] for ind, cand in enumerate(self.alts)}
        self.P = P
        self.create_rank_dicts(alt_scores)
        return gamma, (t1-t0), (t2-t1)


def main():
    print("Executing Unit Tests")
    cand_set = [0, 1, 2]

    print("Testing GMMPL")

    gmmagg = GMMPLAggregator(cand_set)
    # from the paper
    votes = [[0, 1, 2], [1, 2, 0]]
    gmmagg.aggregate(votes)
    #print(gmmagg.P)
    print(gmmagg.alts_to_ranks, gmmagg.ranks_to_alts)
    assert([gmmagg.get_ranking(i) for i in cand_set] == [1,0,2])
    #assert(np.array_equal(gmmagg.P,np.array([[-1,.5,.5],[.5,-.5,1],[.5,0,-1.5]])))

    gmmagg.aggregate(votes, breaking='top', k=2)
    #print(gmmagg.P)
    print(gmmagg.alts_to_ranks, gmmagg.ranks_to_alts)
    print("Tests passed")

if __name__ == "__main__":
    main()
