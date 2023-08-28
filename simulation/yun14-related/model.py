from typing import List, Set, Tuple
from itertools import product
import numpy as np
import random


class Pair:
    left: int
    right: int

    def __init__(self, left: int, right: int):
        self.left = left
        self.right = right


class Model:
    # number of clusters
    K: int
    # number of nodes
    n: int
    # relative size of clusters
    alpha: List[float]
    # probability of observation inter and intra cluster
    cross_prob_mat: np.ndarray
    # a partition of nodes into K mutually disjoint clusters
    V: List[Set[int]]
    # probability of observation between two nodes
    # pair_prob_mat: np.ndarray
    vommitted: List[np.ndarray]
    observed_times: dict

    def __init__(
        self,
        K: int,
        n: int,
        alpha: List[float],
        # cross_prob_mat: np.ndarray,
    ):
        self.K = K
        self.n = n
        self.alpha = alpha
        self.vommitted = []
        self.observed_times = {}
        # self.cross_prob_mat = cross_prob_mat

        assert K >= 2
        assert n > 0
        assert len(alpha) == K
        # assert np.all(
        #     self.cross_prob_mat >= 0) & np.all(
        #     self.cross_prob_mat <= 1)

        self.V = []

        start_idx = 0
        for a in alpha:
            size = int(a * n)
            end_idx = start_idx + size
            self.V.append(set(range(start_idx, end_idx)))
            start_idx = end_idx

        self.pair_prob_mat = np.eye(self.n)
        for i, j in product(range(n), range(n)):
            if i <= j:
                continue
            if self.cluster_of(i) == self.cluster_of(j):
                self.pair_prob_mat[i, j] = np.random.uniform(0.6, 0.85)
            else:
                # self.pair_prob_mat[i, j] = 1e-3 / 20
                self.pair_prob_mat[i, j] = np.random.uniform(0.1, 0.35)
            self.pair_prob_mat[j, i] = self.pair_prob_mat[i, j]

    def sample(self, space=None) -> Pair:
        if space is None:
            space = range(self.n)
        left, right = random.choices(space, k=2)
        return Pair(left, right)

    def cluster_of(self, node: int) -> int:
        for k, v_k in enumerate(self.V):
            if node in v_k:
                return k
        raise ValueError('This should not happen')

    def is_same_cluster(self, pair: Pair) -> bool:
        clusters = [self.cluster_of(pair.left),
                    self.cluster_of(pair.right)]
        return np.all(clusters != -1) and clusters[0] == clusters[1]

    def observe(self, space=None) -> Tuple[int, Pair]:
        pair = self.sample(space)
        prob = self.pair_prob_mat[pair.left, pair.right]
        return 1 if np.random.uniform() < prob else 0, pair

    def vomit(self, T: int, space=None, symmetric=False, ones_only=False) -> np.ndarray:
        obs_mat = np.zeros((self.n, self.n), dtype=int)
        for _ in range(T):
            x, pair = self.observe(space)
            _pair = (pair.left, pair.right)
            if _pair in self.observed_times:
                self.observed_times[_pair] += 1
            else:
                self.observed_times[_pair] = 1
            if ones_only:
                x = -1 if x == 0 else 1
            v, w = pair.left, pair.right
            obs_mat[v, w] += x
            if symmetric:
                obs_mat[w, v] += x

        if ones_only:
            obs_mat[x > 0] = 1
            obs_mat[x < 0] = 0

        self.vommitted.append(obs_mat)
        return obs_mat
