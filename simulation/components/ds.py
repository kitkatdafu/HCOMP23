from __future__ import annotations

import copy
import collections.abc

import random
import pandas as pd
import scipy.io
import numpy as np
import components.metric
import components.funcs
from typing import Set, Hashable, List, Callable, Optional, Collection, Dict,\
    Any
import components.iter


class Vertex(collections.abc.Hashable):
    """
    A vertex in the dataset
    """
    value: Hashable
    true_cluster_id: int
    p: float
    is_always_hard: bool
    predicted_cluster: Optional[Cluster]
    visited_clusters: Set

    def __init__(self,
                 value: Hashable,
                 true_cluster_id: int,
                 p: float,
                 is_always_hard: bool):
        """
        Initialize a vertex
        Args:
            value (Hashable): value corresponds to the vertex
            true_cluster_id (int): vertex's true cluster id
            p (float): probability a *correct* decision is made when this
                vertex is paired with another vertex
            is_always_hard (bool): whether this vertex is a hard vertex or not
        """
        self.value = value
        self.true_cluster_id = true_cluster_id
        self.p = p
        self.is_always_hard = is_always_hard
        self.predicted_cluster = None
        self.visited_clusters = set()

    def set_predicted_cluster(self, predicted_cluster: Cluster):
        """
        Set predicted cluster for this vertex
        Args:
            predicted_cluster (Cluster): set predicted cluster to this vertex
        """
        self.predicted_cluster = predicted_cluster

    def has_visited(self, cluster: Cluster) -> bool:
        """
        Check if this vertex has visited the given cluster or not
        Args:
            cluster (Cluster): a cluster to be checked
        Returns:
            True if this cluster has been visited
        """
        return cluster in self.visited_clusters

    def visit(self, cluster: Cluster):
        """
        Add cluster to the visited set
        """
        self.visited_clusters.add(cluster)

    def __hash__(self) -> int:
        return hash(self.value)

    def __eq__(self, other) -> bool:
        return self.value == other.value

    def __repr__(self):
        return str((self.value, self.true_cluster_id, self.is_always_hard))


class Pair(collections.abc.Hashable):
    left: Vertex
    right: Cluster

    def __init__(self, left: Vertex, right: Cluster):
        self.left = left
        self.right = right

    def from_same_true_cluster(self) -> bool:
        """
        Check if left and the rep of right has the same true cluster id.
        This function may sample a new rep according to the hyperparameter of
        the algorithm.
        Returns: True if same
        """
        return self.rep_and_left_same_true_cluster()

    def rep_and_left_same_true_cluster(self) -> bool:
        """
        Check if left and the rep of right has the same true cluster id
        Returns: True if same
        """
        return self.left.true_cluster_id == self.right.rep.true_cluster_id

    def __hash__(self):
        return hash((self.left, self.right))

    def __eq__(self, other):
        return self.left == other.left and self.right == other.right

    def __repr__(self):
        return str((self.left, self.right))


class Batch(collections.abc.Collection):
    """
    A batch of vertices
    """
    batch: Set[Vertex]
    batch_capacity: int

    def __init__(self, batch_capacity: int):
        """
        Initialize a batch
        Args:
            batch_capacity (int): capacity of the batch
        """
        self.batch = set()
        self.batch_capacity = batch_capacity

    def fill(self, dataset: Dataset, rng: np.random.Generator) -> List[Vertex]:
        """
        Fill the as full as possible using vertices from the given from the
        given dataset. Note that vertices added to the batch should be
        removed from the dataset. This function does not handle this step.
        Callers of this function should handle this themselves
        Args:
            dataset (Dataset): dataset
            rng (np.random.Generator): random generator that is used solely
            to fill batch
        Returns: a list of vertices added to the batch
        """

        # if the dataset is empty, do nothing
        if len(dataset) == 0:
            return []

        sample_size = min(self.batch_capacity - len(self.batch), len(dataset))
        # vertices in dataset that will be added to the batch
        vertices = list(rng.choice(dataset, size=sample_size, replace=False))
        self.batch.update(vertices)
        return vertices

    def remove(self, vertex: Vertex):
        """
        Remove a vertex from this batch
        Args:
            vertex (Vertex): a vertex to be removed
        """
        self.batch.remove(vertex)

    def form_cluster(self, parameters: Parameters) -> Cluster:
        """
        Pick and remove a vertex from the batch uniformly at random. Create a
        singleton cluster using this vertex
        Returns: A singleton cluster
        """
        pivot = random.sample(self.batch, 1)[0]
        # print("pivot:", pivot)
        self.batch.remove(pivot)
        return Cluster.singleton(pivot, parameters)

    def __contains__(self, item):
        pass

    def __iter__(self):
        return iter(self.batch)

    def __len__(self):
        return len(self.batch)

    def __repr__(self):
        return "Batch: " + str(self.batch)


class Cluster(collections.abc.Set, Hashable):
    """
    A cluster formed by the algorithm
    """

    cluster: Set[Vertex]
    enable_random_rep: bool
    rep: Optional[Vertex]

    def __init__(self, enable_random_rep: bool):
        """
        Initialize a cluster
        """
        self.cluster = set()
        self.rep = None
        self.enable_random_rep = enable_random_rep

    def pick_rep(self, ):
        """
        Pick a rep (again)
        """
        self.rep = random.sample(self.cluster, 1)[0]

    def add(self, vertex: Vertex):
        """
        Add a vertex into this cluster
        Args:
            vertex (Vertex): vertex to be added
        """
        self.cluster.add(vertex)

    def remove(self, vertex: Vertex):
        """
        Remove a vertex from this cluster
        Args:
            vertex (Vertex): vertex to be removed
        """
        self.cluster.remove(vertex)

    @staticmethod
    def singleton(vertex: Vertex, parameters: Parameters) -> Cluster:
        """
        Create a singleton cluster that contains the given vertex
        Args:
            vertex (Vertex): the only vertex that is going to exists in the
            parameters (Parameters): parameters used in the algorithm
            cluster returned from this function
        Returns:
            A singleton cluster
        """
        cluster = Cluster(parameters.enable_random_rep)
        cluster.add(vertex)
        return cluster

    @staticmethod
    def dummy(vertices: Collection[Vertex]) -> Cluster:
        """
        Create a dummy cluster that contains the given vertices
        Args:
            vertices: given vertices

        Returns:
            a dummy cluster
        """
        cluster = Cluster(False)
        cluster.cluster.update(vertices)
        return cluster

    def __contains__(self, item: Vertex):
        return item in self.cluster

    def __len__(self):
        return len(self.cluster)

    def __iter__(self):
        return iter(self.cluster)

    def __hash__(self):
        return hash(id(self))

    def __repr__(self):
        return str(self.cluster)

    def __eq__(self, other):
        return id(other) == id(self)


class Clusters(collections.abc.Sequence):
    """
    A collection of clusters
    """

    clusters: List[Cluster]

    def __init__(self):
        """
        Initialize a clusters (list of cluster)
        """
        self.clusters = []

    def add(self, cluster: Cluster):
        """
        Add a cluster into clusters
        Args:
            cluster (Cluster): cluster to be added
        """
        self.clusters.append(cluster)

    def remove(self, cluster: Cluster):
        """
        Remove a cluster from clusters
        Args:
            cluster (Cluster): cluster to be removed
        """
        self.clusters.remove(cluster)

    def __len__(self):
        return len(self.clusters)

    def __getitem__(self, item):
        return self.clusters[item]

    def __repr__(self):
        return str(self.clusters)


class Dataset(collections.abc.Sequence):
    """
    Input to the algorithm
    """
    n: int
    c: int
    p: float
    always_hard_ratio: float
    always_hard_p: float
    dataset: List[Vertex]

    def __init__(self,
                 dataset: List[Vertex],
                 c: int,
                 n: int,
                 p: float,
                 always_hard_p: float,
                 always_hard_ratio: float):
        """
        Initialize a dataset
        Args:
            dataset (List[Vertex]): a collection of vertices
        """
        self.dataset = dataset
        self.c = c
        self.n = n
        self.p = p
        self.always_hard_ratio = always_hard_ratio
        self.always_hard_p = always_hard_p

    @staticmethod
    def all_sports(all_sports_file_name: str):
        data = scipy.io.loadmat(all_sports_file_name)
        vertices = []
        for cluster_id, cluster in enumerate(data["trueC"]):
            for value in cluster[0][0]:
                vertices.append(Vertex(value,
                                       cluster_id,
                                       -1,
                                       False))
        return Dataset(dataset=vertices,
                       c=len(data["trueC"]),
                       n=len(vertices),
                       p=-1,
                       always_hard_p=-1,
                       always_hard_ratio=-1)

    @staticmethod
    def yun_14(K: int):
        vertices = []
        for cluster_id, cluster in enumerate([
                range(
                    k * int(900 / K),
                    (k + 1) * int(900 / K))
                for k in range(K)]):
            for value in cluster:
                vertices.append(Vertex(value,
                                       cluster_id,
                                       -1,
                                       False))
        return Dataset(dataset=vertices,
                       c=K,
                       n=900,
                       p=-1,
                       always_hard_p=-1,
                       always_hard_ratio=-1)

    def remove(self, vertex: Vertex):
        """
        Remove a given vertex from the dataset
        Args:
            vertex (Vertex): a vertex to be removed
        """
        self.dataset.remove(vertex)

    def remove_many(self, vertices: List[Vertex]):
        """
        Remove a list of vertices form the dataset
        Args:
            vertices (List[Vertex]): a list of vertices to be removed
        """
        for vertex in vertices:
            self.remove(vertex)

    def true_clusters(self, enable_random_rep: bool) -> components.ds.Clusters:
        """
        Generate the true clusters from dataset.
        Returns:
            True clusters
        """
        true_clusters = components.ds.Clusters()
        for _ in range(self.c):
            true_clusters.add(components.ds.Cluster(enable_random_rep))
        for vertex in self:
            true_clusters[vertex.true_cluster_id].add(vertex)
        return true_clusters

    @staticmethod
    def dummy(n: int,
              c: int,
              p: float,
              always_hard_p: float,
              always_hard_ratio: float):
        """
        Generate a dataset that has n vertices such that these n vertices
        are partitioned into c clusters.
        Args:
            n (int): number of vertices
            c (int): number of clusters
            p (float): probability that a pair of vertices is correctly
                clustered
            always_hard_p (float): probability that a pair of vertices is
                correctly clustered when the pair *contains* a hard vertex
            always_hard_ratio (float): ratio of number of hard
                vertices to the size of the dataset

        Returns: Generated dummy dataset
        """
        if n <= 0:
            raise ValueError("n should be at least 1")
        if c <= 0:
            raise ValueError("c should be at least 1")
        if c > n:
            raise ValueError("c should not be greater than n")
        if p <= 0.5 or p > 1:
            raise ValueError("p should be greater than 0.5 and less than "
                             "or equal to 1")
        if always_hard_p <= 0.5 or always_hard_p > 1:
            raise ValueError("always_hard_p should be greater than 0.5 and "
                             "less than or equal to 1")
        if always_hard_ratio < 0 or always_hard_ratio > 1:
            raise ValueError("always_hard_ratio should be greater than or "
                             "equal to 0 and less than or equal to 1")

        dataset = []
        for i in range(n):
            if i >= int(always_hard_ratio * n):
                dataset.append(Vertex(i, i % c, p, False))
            else:
                dataset.append(Vertex(i, i % c, always_hard_p, True))

        return Dataset(dataset,
                       c=c,
                       n=n,
                       p=p,
                       always_hard_p=always_hard_p,
                       always_hard_ratio=always_hard_ratio)

    @staticmethod
    def from_json(data):
        value_to_vertex = {}
        vertices = []
        c = set()
        for vertex_data in data:
            vertex = Vertex(value=vertex_data["value"],
                            true_cluster_id=vertex_data["true_cluster_id"],
                            p=1,
                            is_always_hard=False)
            c.add(vertex_data["true_cluster_id"])
            value_to_vertex[vertex_data["value"]] = vertex
            vertices.append(vertex)
        import math
        return Dataset(dataset=vertices,
                       n=len(vertices),
                       c=len(c),
                       p=math.nan,
                       always_hard_p=math.nan,
                       always_hard_ratio=math.nan), value_to_vertex

    def __repr__(self):
        return str(self.dataset)

    def __contains__(self, item):
        return item in self.dataset

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]


class Parameters:
    """
    Hyperparameters that control the behavior of the algorithm
    """
    batch_capacity: int
    max_workers: int
    cap: int
    phi_function: Callable
    eta: float
    delta: float
    enable_random_rep: bool

    def __init__(self, **kwargs):
        self.batch_capacity = kwargs.get('batch_capacity')
        self.max_workers = kwargs.get('max_workers')
        self.cap = kwargs.get('cap')
        self.phi_function = kwargs.get('phi_function')
        self.eta = kwargs.get('eta')
        self.delta = kwargs.get('delta')
        self.enable_random_rep = kwargs.get('enable_random_rep')

    @staticmethod
    def str_to_phi_func(name: str) -> Callable:
        if name == "default":
            return components.funcs.default_phi_function
        else:
            raise ValueError("No such phi function")

    @staticmethod
    def from_json(data) -> Parameters:
        """
        Load parameters from json object
        Args:
            data: data to be converted to Parameters
        Returns:
            An Parameters object
        """
        parameters = Parameters(batch_capacity=data["batch_capacity"],
                                max_workers=data["max_workers"],
                                enable_random_rep=data["enable_random_rep"],
                                cap=data["cap"],
                                phi_function=Parameters.str_to_phi_func(
                                    data["phi_func"]),
                                eta=data["eta"],
                                delta=data["delta"])

        return parameters

    def __repr__(self):
        text = "Parameters:\n"
        for k, v in self.__dict__.items():
            text += "\t{}: {}\n".format(k, v)
        return text


class Tracker:
    """
    A tracker that tracks the statistics of the algorithm
    """
    dataset: Dataset
    parameters: Parameters
    T: Dict[Pair, int]
    n_queries: int
    n_edges: int
    n_decisions: int
    n_correct_edges: int
    n_correct_decisions: int
    n_encounter_always_hard: int
    c: int
    n: int
    p: float
    always_hard_p: float
    always_hard_ratio: float
    enable_random_rep: bool
    batch_capacity: int
    max_workers: int
    hard: Set[Pair]
    vi_full: Optional[float]
    vi_clustered: Optional[float]
    min_T: Optional[int]
    max_T: Optional[int]
    mode_T: Optional[int]
    median_T: Optional[int]
    mean_T: Optional[float]
    sd_T: Optional[float]
    num_hards: Optional[int]
    decision_error_rate: Optional[float]
    edge_error_rate: Optional[float]
    num_clusters: Optional[int]
    draw_tracking: Optional[List[List[Vertex]]]
    size_of_each_cluster: List[int]

    @staticmethod
    def from_json(json_data):
        parameters = Parameters.from_json(json_data["parameters"])

        dataset, value_to_vertex = Dataset.from_json(json_data[
            "dataset_bak"])

        tracker = Tracker(dataset, parameters)
        tracker.n_queries = json_data["n_queries"]
        tracker.n_edges = json_data["n_edges"]
        tracker.n_correct_edges = json_data["n_correct_edges"]
        tracker.n_decisions = json_data["n_decisions"]
        tracker.n_correct_decisions = json_data["n_correct_decisions"]

        # handle T
        for key, t_count in json_data["T"].items():
            tracker.T[key] = t_count

        # handle hard
        for hard_value in json_data["hard"]:
            left_value = hard_value
            # values = hard_value.split(",")
            # left_value = values[0].strip()
            #  WARNING
            # right_value = values[1].strip()
            tracker.hard.add(Pair(value_to_vertex[left_value],
                                  Cluster.singleton(left_value,
                                                    parameters)))
        # handle clusters
        clusters = Clusters()
        for cluster_data in json_data["clusters"]:
            cluster = Cluster(parameters.enable_random_rep)
            for vertex_data in cluster_data:
                vertex = value_to_vertex[vertex_data["value"]]
                cluster.add(vertex)
            clusters.add(cluster)

        # tracker.wrap_up(clusters, json_data["draw_tracking"])
        tracker.wrap_up(clusters, [])
        return tracker, value_to_vertex

    def __init__(self, dataset: Dataset, parameters: Parameters):
        """
        Initialize a tracker
        Args:
            dataset  (Dataset):  dataset
        """
        self.T = {}  #
        self.n_queries = 0
        self.n_edges = 0
        self.n_decisions = 0
        self.n_correct_edges = 0
        self.n_correct_decisions = 0
        self.n_encounter_always_hard = 0
        self.vi_full = None
        self.vi_clustered = None
        self.dataset = copy.deepcopy(dataset)
        self.parameters = parameters
        self.enable_random_rep = parameters.enable_random_rep
        self.eta = parameters.eta
        self.delta = parameters.delta
        self.cap = parameters.cap
        self.batch_capacity = parameters.batch_capacity
        self.max_workers = parameters.max_workers
        self.n = dataset.n
        self.c = dataset.c
        self.p = dataset.p
        self.always_hard_p = dataset.always_hard_p
        self.always_hard_ratio = dataset.always_hard_ratio
        self.hard = set()
        self.draw_tracking = None

    def param_id(self):
        return (self.n,
                self.cap,
                self.eta,
                self.delta,
                self.c,
                self.p,
                self.always_hard_p,
                self.always_hard_ratio,
                self.batch_capacity,
                self.max_workers)

    def increment_n_queries(self):
        """
        Add 1 to n_queries
        """
        self.n_queries += 1

    def increment_T_at_pair(self, pair: Pair):
        """
        Add 1 to T[pair]
        Args:
            pair (pair): a pair that has been queried
        """
        self.T[pair] = self.T.get(pair, 0) + 1

    def same_cluster(self, pair: Pair):
        """
        The given pair has been determined as having the same cluster. This
        function first checks if the decision has been made correctly,
        and then update statistics accordingly
        Args:
            pair (Pair): a pair that has been determined as having same cluster
        """
        if pair.rep_and_left_same_true_cluster():
            self.n_correct_edges += 1
            self.n_correct_decisions += 1
        self.n_edges += 1
        self.n_decisions += 1

    def diff_cluster(self, pair: Pair):
        """
        The given pair has been determined as having different clusters. This
        function first checks if the decision has been made correctly,
        and then update statistics accordingly
        Args:
            pair (Pair): a pair that has been determined as having
                different clusters
        """
        if not pair.rep_and_left_same_true_cluster():
            self.n_correct_decisions += 1
        self.n_decisions += 1

    def add_hard(self, pair: Pair):
        """
        The given pair has been determined as hard. This pair will be added
        to the hard_set

        Args:
            pair (Pair): a pair that has been determined as hard
        """
        self.hard.add(pair)

    def is_hard_pair(self, pair: Pair):
        """
        Check if pair is in hard set
        Args:
            pair: a pair to be checked
        """
        return pair in self.hard

    def increment_n_encounter_always_hard(self):
        """
        Increment n encounter always hard
        """
        self.n_encounter_always_hard += 1

    def calculate_voi(self, clusters: Clusters):
        """
        Calculate variation of information
        Args:
            clusters (Clusters): predicted clusters
        """
        clusters_full = copy.deepcopy(clusters)
        clusters_full.add(Cluster.dummy([hard_vertex.left
                                         for hard_vertex in self.hard]))
        self.vi_full = components.metric.voi(clusters_full,
                                             self.dataset)
        self.vi_clustered = components.metric.voi(clusters,
                                                  self.dataset)

    def wrap_up(self, clusters: Clusters, draw_tracking: List[List[Vertex]]):
        """
        Calculate all statistics. This function should be called at the end
        of execute()
        Args:
            clusters (Clusters): predicted clusters
        """
        import statistics
        self.calculate_voi(clusters)
        self.min_T = min(self.T.values())
        self.max_T = max(self.T.values())
        self.median_T = statistics.median(self.T.values())
        self.mean_T = statistics.mean(self.T.values())
        self.sd_T = statistics.stdev(self.T.values())
        self.mode_T = statistics.mode(self.T.values())
        self.num_hards = len(self.hard)
        self.num_clusters = len(clusters)
        self.draw_tracking = draw_tracking
        self.size_of_each_cluster = list(map(lambda x: len(x), clusters))

        try:
            self.decision_error_rate = 1 - self.n_correct_decisions / \
                self.n_decisions
        except ZeroDivisionError:
            self.decision_error_rate = np.nan

        try:
            self.edge_error_rate = 1 - self.n_correct_edges / self.n_edges
        except ZeroDivisionError:
            self.edge_error_rate = np.nan

    def as_pd(self) -> pd.Series:
        """
        Convert tracking data into pandas series
        Returns:
            pd.Series representation of the tracker
        """
        di = self.__dict__
        di["T"] = list(di["T"].values())
        del di["dataset"]
        del di["hard"]
        del di["draw_tracking"]
        del di["parameters"]
        return pd.Series(di)
