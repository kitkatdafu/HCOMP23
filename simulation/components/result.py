from __future__ import annotations
import components.ds

import enum


class ClusterResultHard:
    def __init__(self):
        pass


class ClusterResultUndetermined:
    def __init__(self):
        pass


class ClusterResultSameCluster:
    def __init__(self, cluster: components.ds.Cluster):
        self.cluster = cluster


class QueryResult(enum.IntEnum):
    NOT_SAME = 0
    SAME = 1
