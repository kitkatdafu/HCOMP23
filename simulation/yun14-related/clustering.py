import numpy as np
from itertools import product, combinations
from random import choice, choices, sample
from math import ceil, prod
from typing import List, Set
from model import Model


def find_K(obs_mat: np.ndarray) -> int:
    one_obs_mat = np.copy(obs_mat)
    # one_obs_mat[one_obs_mat > 0] = 1

    n = len(obs_mat)

    U, S, Vt = np.linalg.svd(one_obs_mat)
    p_hat = np.sum(one_obs_mat > 0) / np.prod(one_obs_mat.shape)
    eta = 0.01
    K = len([s for s in S if s >= (2 + eta) * np.sqrt(n * p_hat)])
    return K


def trimming(obs_mat: np.ndarray, K: int) -> (np.ndarray, np.ndarray):
    '''
    Trim the observation matrix A.

    Let Tau be a set of nodes that did not get too many positive observations, i.e.
    Tau = {v | sum(A[v, w], for all w) <= 10 * sum(A[v, w], for all (v, w)) / n}

    We keep the entries corresponding to Tau
    '''

    n = len(obs_mat)
    bound = 5 * K * obs_mat.sum() / n
    tau = np.array([i for i in range(n) if obs_mat[:, i].sum() <= bound])

    trimmed_obs_mat = np.zeros((n, n), dtype=int)
    for v, w in product(tau, tau):
        trimmed_obs_mat[v, w] = obs_mat[v, w]

    return trimmed_obs_mat, tau


def spectral_decomposition(trimmed_obs_mat: np.ndarray,
                           K: int,
                           bound: float,
                           tau: np.ndarray) -> List[Set[int]]:
    '''
    Extract the clusters from the spectral analysis of trimmed A

    Let K denote the rank of the trimmed A.
    Extract the clusters from the column vectors of the rank-K approximation matrix approx_obs_hat of trimmed A.
    The rank-K approximation is done by SVD.

    Columns vectors corresponding to nodes in the same clusters should be relatively closer to each other.
    '''

    n = len(trimmed_obs_mat)

    U, S, Vt = np.linalg.svd(trimmed_obs_mat)
    S[K:] = 0
    approx_obs_mat = U @ np.diag(S) @ Vt

    best = (np.inf, [])

    for i in range(int(np.log(n))):

        Q = []
        for v in range(n):
            Q_v = set()
            for w in range(n):
                dist = np.linalg.norm(
                    approx_obs_mat[:, w] - approx_obs_mat[:, v]) ** 2
                if dist > (i + 1) * bound / (100 * n ** 2):
                    continue
                Q_v.add(w)
            Q.append(Q_v)

        T = []
        xi = []
        for k in range(K):
            v_star = np.argmax([len(Q[v] - set.union(*T[:k], set()))
                               for v in range(n)])
            T.append(Q[v_star] - set.union(*T[:k], set()))
            xi.append(np.sum([approx_obs_mat[:, v] / len(T[k])
                              for v in T[k]],
                             axis=0))

        for v in set(tau) - set.union(*T):
            k_star = np.argmin(
                [np.linalg.norm(approx_obs_mat[:, v] - xi[k]) ** 2
                 for k in range(K)])

            T[k_star].add(v)

        r = sum([np.linalg.norm(approx_obs_mat[:, v] - xi[k]) ** 2
                 for v in T[k]
                 for k in range(K)])

        best = min(best, (r, T))

    return best[1]


def improvement(obs_mat: np.ndarray,
                clusters: List[Set[int]],
                n: int) -> List[Set[int]]:
    K = len(clusters)

    for i in range(int(np.log(n))):
        improved_clusters = [set() for _ in range(K)]
        for v in range(n):
            prop = np.array(
                [np.sum([obs_mat[v, w] for w in clusters[k]]) /
                 len(clusters[k]) for k in range(K)])
            np.nan_to_num(prop, nan=-1, copy=False)
            k_star = np.random.choice(np.where(prop == prop.max())[0])
            improved_clusters[k_star].add(v)
        clusters = improved_clusters

    return improved_clusters


def spectral_partition(obs_mat: np.ndarray):
    '''
    A clustering algorithm for non-adaptive URS-1 or URS-2 sampling strategies.
    From the T observations, we construct an n by n matrix A where

    A[v, w] is equal to the number of positive observation sof node pair (v, w).
    A[v, w] = 0 if the pair (v, w) has not been observed.

    If v and w are in the same cluster, E[A[v, w]] = 2 * T / (n * (n - 1)) * p
    If v and w are not in the same cluster, E[A[v, w]] = 2 * T / (n * (n - 1)) * q

    Matrix E[A] is symmetric and of rank K

    We expect to accurately recover the clusters if 
    (p - q) ** 2 / (p + q) * T / n >> 1

    1. Trimming
    2. Spectral Decomposition
    3. Improvement
    '''
    n = len(obs_mat)
    K = find_K(obs_mat)

    # print('Trimming...')
    trimmed_obs_mat, tau = trimming(obs_mat, K)

    # print('Spectral Decomposing...')
    bound = obs_mat.sum() / n ** 2
    clusters = spectral_decomposition(trimmed_obs_mat, K, bound, tau)

    # print('Improving...')
    improved_clusters = improvement(obs_mat, clusters, n)

    return improved_clusters


def adaptive_spectral_partition(T: int,
                                num_nodes_kernel: int,
                                model: Model):
    T_const = T
    # print('Initializing...')
    n = model.n
    # nodes not in the reference kernel
    nodes = set(range(n))

    # nodes in the reference kernel
    S = set(sample(range(n), k=num_nodes_kernel))
    nodes = nodes - S
    S = sorted(list(S))

    # construct an observation matrix
    T_S = int(T / 5)
    kernel_obs_mat = model.vomit(T_S, S)
    T -= T_S

    # print('Finding the reference kernels...')
    clusters = spectral_partition(kernel_obs_mat[S][:, S])
    revised_clusters = []
    for cluster in clusters:
        revised_clusters.append(set(map(lambda x: S[x], cluster)))
    from copy import deepcopy
    fixed_clusters = deepcopy(revised_clusters)
    K = len(clusters)

    # Estimate p and q
    # print('Estimating p and q...')
    cst = (len(S) ** 2) / (2 * T_const)
    sum_squared_cart = sum(map(lambda x: len(x) ** 2, clusters))

    numerator = 0
    for cluster in map(list, clusters):
        numerator += kernel_obs_mat[S][:, S][cluster][:, cluster].sum()
    p_hat = numerator / sum_squared_cart * cst
    # print(f'\t{clusters = }')

    numerator = 0
    for cluster in map(list, clusters):
        cluster_c = list(set(range(len(S))) - set(cluster))
        numerator += kernel_obs_mat[S][:, S][cluster][:, cluster_c].sum()
    q_hat = numerator / (len(S) ** 2 - sum_squared_cart) * cst

    pair_times = {}
    budget_overflowed_nodes = []
    # Classify the remaining nodes
    # print('Classiyfing the remaining nodes')
    while len(nodes) > 0 and T > 0:
        nodes_to_be_removed = []
        for v in nodes:
            A = []
            for cluster in fixed_clusters:
                if len(cluster) == 0:
                    continue
                size = ceil(2 * T_const / (3 * K * n))
                T -= size
                chosen_nodes = choices(list(cluster), k=size)
                pair_times[v] = pair_times.get(v, 0) + size
                prob = np.array([model.pair_prob_mat[v, w]
                                for w in chosen_nodes])
                A.append((np.random.uniform(size=size) < prob).sum())

            k_star = np.argmax(A)
            d_star = np.inf
            for k, a in enumerate(A):
                if k == k_star:
                    continue
                d_star = min(d_star, A[k_star] - a)

            if d_star >= (p_hat - q_hat) / (2 * K) * T_const / n:
                if T <= 0:
                    budget_overflowed_nodes.append(v)
                revised_clusters[k_star].add(v)
                nodes_to_be_removed.append(v)

        for v in nodes_to_be_removed:
            nodes.remove(v)

    for v in nodes:
        k = np.random.randint(K)
        revised_clusters[k].add(v)

    return revised_clusters, budget_overflowed_nodes, T, p_hat, q_hat, pair_times
