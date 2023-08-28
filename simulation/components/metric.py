from __future__ import annotations

import components.ds
from typing import Dict, Set
import numpy as np


def voi(clusters: components.ds.Clusters,
        dataset: components.ds.Dataset) -> float:
    """
    Calculate the variation of information between predicted clusters (
    clusters) and the true clusters (dataset)
    If the number of vertices in clusters does not equal to the number of
    vertices in the dataset, the VI is calculated on the vertices presented
    in the clusters.

    Args:
        clusters (Clusters): predicted clusters
        dataset (Dataset): dataset (contains the true clusters)
    Returns:
        variation of information
    """
    set_of_vertices = set()
    [set_of_vertices.update(cluster.cluster) for cluster in clusters]

    true_clusters_dict: Dict[int, Set[components.ds.Vertex]] = {}
    for vertex in dataset:
        if vertex not in set_of_vertices:
            continue
        true_cluster_id = vertex.true_cluster_id
        if true_cluster_id in true_clusters_dict:
            true_clusters_dict[true_cluster_id].add(vertex)
        else:
            true_clusters_dict[true_cluster_id] = {vertex}

    true_clusters = list(true_clusters_dict.values())
    predicted_clusters = [cluster.cluster for cluster in clusters]

    n = len(dataset)
    p = np.array([len(cluster) / n for cluster in true_clusters])
    q = np.array([len(cluster) / n for cluster in predicted_clusters])

    r = np.zeros((len(true_clusters), len(predicted_clusters)))
    for i in range(len(true_clusters)):
        for j in range(len(predicted_clusters)):
            r[i, j] = len(
                true_clusters[i].intersection(predicted_clusters[j])) / n

    vi: float = 0
    for i in range(len(true_clusters)):
        for j in range(len(predicted_clusters)):
            if r[i, j] <= 0:
                continue
            vi += r[i, j] * ((np.log2(r[i, j]) - np.log2(p[i]))
                             + (np.log2(r[i, j]) - np.log2(q[j])))

    return abs(vi)