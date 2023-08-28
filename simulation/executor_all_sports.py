from __future__ import annotations
from typing import Tuple, Final, Union, List, Dict

import components.ds as ds
import components.result
import components.funcs
import components.variable
import concurrent.futures as cf

import random
import numpy as np
import scipy.io
import pandas as pd


class BatchQueryExecutor:
    dataset: ds.Dataset
    parameters: Final[ds.Parameters]
    batch: ds.Batch
    tracker: ds.Tracker
    clusters: ds.Clusters
    # trace the order of how vertices are sampled from the dataset and are
    # added to the batch
    draw_tracking: List[List[ds.Vertex]]
    existed_answers: Dict[Tuple, List[int]]

    def __init__(self,
                 dataset: ds.Dataset,
                 parameters: ds.Parameters,
                 existed_answers: Dict[Tuple, List[int]],
                 conf):
        """
        Initialize a batch query executor
        Args:
            dataset (Dataset): dataset to be clustered
            parameters (Parameters): parameters to be used
        """
        self.dataset = dataset
        self.parameters = parameters
        self.batch = ds.Batch(parameters.batch_capacity)
        self.tracker = ds.Tracker(dataset, parameters)
        self.clusters = ds.Clusters()
        self.draw_tracking = []
        self.existed_answers = existed_answers
        self.conf = conf

    def is_executable(self) -> bool:
        """
        Determine if the executor can execute or not, i.e. if there there are
        still vertices left to be clustered
        Returns: True if this executor can further execute
        """
        return len(self.dataset) + len(self.batch) != 0

    def execute(self, seed=853):
        """
        Execute the algorithm
        """
        batch_rng = np.random.default_rng(seed)
        with cf.ThreadPoolExecutor() as pe:
            while self.is_executable():
                future_results = [pe.submit(self.cluster, vertex)
                                  for vertex in self.batch]
                for future_result in cf.as_completed(future_results):
                    vertex, result = future_result.result()
                    if not isinstance(
                            result, components.result.ClusterResultUndetermined):
                        self.batch.remove(vertex)
                        if not isinstance(result,
                                          components.result.ClusterResultHard):
                            result.cluster.add(vertex)
                            vertex.set_predicted_cluster(result.cluster)
                if len(self.batch) != 0:
                    cluster = self.batch.form_cluster(self.parameters)
                    self.clusters.add(cluster)
                vertices = self.batch.fill(self.dataset, batch_rng)
                self.draw_tracking.append(vertices)
                self.dataset.remove_many(vertices)

            self.tracker.wrap_up(self.clusters, self.draw_tracking)
        return self.tracker

    def query(self,
              pair: ds.Pair) -> Tuple[components.result.QueryResult,
                                      ds.Pair]:
        import random
        left = pair.left.value
        right = pair.right.rep.value

        if (left, right) in self.existed_answers:
            ans = random.sample(self.existed_answers[(left, right)], 1)[0]
            if ans == 1:
                return components.result.QueryResult.SAME, pair
            else:
                return components.result.QueryResult.NOT_SAME, pair
        else:
            rd = random.random()
            p = self.conf[left - 1, right - 1]
            if rd < p:
                return components.result.QueryResult.SAME, pair
            else:
                return components.result.QueryResult.NOT_SAME, pair

    def cluster(self, vertex: ds.Vertex) -> Tuple[
        ds.Vertex,
        Union[components.result.ClusterResultUndetermined,
              components.result.ClusterResultSameCluster,
              components.result.ClusterResultHard]]:

        with cf.ThreadPoolExecutor() as pe:
            # for each existing cluster
            for cluster in self.clusters:
                # check if the vertex has visited this cluster or not
                if vertex.has_visited(cluster):
                    continue
                cluster.pick_rep()

                step = 1
                cea = components.variable.CumulativeEmpiricalAverage()
                ci = components.variable.ConfidenceInterval(self.parameters)
                done_with_this_cluster = False

                while not done_with_this_cluster:
                    pairs = [ds.Pair(vertex, cluster)
                             for _ in range(self.parameters.max_workers)]

                    future_results = [pe.submit(self.query, pair)
                                      for pair in pairs]

                    for future_result in cf.as_completed(future_results):
                        result, pair = future_result.result()

                        # handle cap
                        if pair in self.tracker.T:
                            if self.tracker.T[pair] >= self.parameters.cap:
                                self.tracker.add_hard(pair)
                                return vertex, components.result.ClusterResultHard()

                        cea.update(step, result)
                        ci.update(step)

                        self.tracker.increment_n_queries()
                        self.tracker.increment_T_at_pair(pair)

                        if cea.lower_confidence_bound(step, ci) > 0.5:
                            self.tracker.same_cluster(pair)
                            return vertex, \
                                components.result.ClusterResultSameCluster(
                                    cluster)
                        elif cea.upper_confidence_bound(step, ci) < 0.5:
                            self.tracker.diff_cluster(pair)
                            vertex.visit(cluster)
                            done_with_this_cluster = True
                            break
                        else:
                            step += 1

        return vertex, components.result.ClusterResultUndetermined()

    def set_warmup_and_execute(self,
                               n_already_clustered_clusters: int,
                               from_hard_ratios: List[float],
                               from_non_hard_ratios: List[float]):
        # create true clusters
        true_clusters = self.dataset.true_clusters(
            self.parameters.enable_random_rep
        )

        # snapped
        snapped_clusters = ds.Clusters()
        [snapped_clusters.add(cluster)
         for cluster in
         random.sample(true_clusters, n_already_clustered_clusters)]

        for i in range(n_already_clustered_clusters):
            cluster = snapped_clusters[i]
            from_hard_ratio = from_hard_ratios[i]
            from_non_hard_ratio = from_non_hard_ratios[i]

            hards = list(filter(lambda x: x.is_always_hard, cluster))
            non_hards = list(filter(lambda x: not x.is_always_hard, cluster))

            # handle hards
            [cluster.remove(vertex)
             for vertex in random.sample(hards,
                                         int(len(hards) *
                                             (1 - from_hard_ratio)))]
            # handle non_hards
            [cluster.remove(vertex)
             for vertex in random.sample(non_hards,
                                         int(len(non_hards) *
                                             (1 - from_non_hard_ratio)))]
        for cluster in snapped_clusters:
            for vertex in cluster:
                self.dataset.remove(vertex)

        # remove empty clusters
        for cluster in list(filter(lambda x: len(x) == 0, snapped_clusters)):
            snapped_clusters.clusters.remove(cluster)
        self.clusters = snapped_clusters

        return self.execute()

    @staticmethod
    def warmup(dataset: ds.Dataset,
               parameters: ds.Parameters,
               n_already_clustered_clusters: int,
               from_non_hard_ratios: List[float],
               from_hard_ratios: List[float],
               ) -> BatchQueryExecutor:
        """
        Warmup an executor from dataset that has been partially clustered.
        How the dataset is partially clustered is determined by the last 3 
        parameters.

        n_already_clustered_clusters, self-evidently, is the number of
        clustered existed before the warmup.
        from_non_had_ratios and from_hard_ratios are both a list of floats.
        The length of both lists should be equivalent to
        n_already_clustered_clusters.

        The i-th float in from_non_hard_ratios (from_hard_ratios) is the
        ratio of non-hard (hard) vertices in the i-th already clustered
        cluster to the number of non-hard (hard) vertices in the total number of
        non-hard (hard) vertices that belong to this
        cluster.

        Args:
            dataset (Dataset): Dataset to be clustered
            parameters (Parameters): Parameters of the algorithm
            n_already_clustered_clusters (int): the number of clustered
                existed before the warmup
            from_non_hard_ratios (List[float]): list of ratios
            from_hard_ratios (List[float]): list of ratios

        Returns:
            A BatchQueryExecutor that has already clustered clusters
        """

        if n_already_clustered_clusters != \
                len(from_non_hard_ratios) != \
                len(from_hard_ratios) != dataset.c:
            raise ValueError("length does not match")

        if len(list(filter(lambda x: x < 0 or x > 1, from_non_hard_ratios))):
            raise ValueError("Non-hard ratios should be >= 0 and <= 1")
        if len(list(filter(lambda x: x < 0 or x > 1, from_hard_ratios))):
            raise ValueError("Non-hard ratios should be >= 0 and <= 1")

        executor = BatchQueryExecutor(dataset, parameters)

        # create true clusters
        true_clusters = dataset.true_clusters(parameters.enable_random_rep)

        # snapped
        snapped_clusters = ds.Clusters()
        [snapped_clusters.add(cluster)
         for cluster in
         random.sample(true_clusters, n_already_clustered_clusters)]

        for i in range(n_already_clustered_clusters):
            cluster = snapped_clusters[i]
            from_hard_ratio = from_hard_ratios[i]
            from_non_hard_ratio = from_non_hard_ratios[i]

            hards = list(filter(lambda x: x.is_always_hard, cluster))
            non_hards = list(filter(lambda x: not x.is_always_hard, cluster))

            # handle hards
            [cluster.remove(vertex)
             for vertex in random.sample(hards,
                                         int(len(hards) *
                                             (1 - from_hard_ratio)))]
            # handle non_hards
            [cluster.remove(vertex)
             for vertex in random.sample(non_hards,
                                         int(len(non_hards) *
                                             (1 - from_non_hard_ratio)))]
        for cluster in snapped_clusters:
            for vertex in cluster:
                dataset.remove(vertex)

        # remove empty clusters
        for cluster in list(filter(lambda x: len(x) == 0, snapped_clusters)):
            snapped_clusters.clusters.remove(cluster)

        executor.clusters = snapped_clusters
        executor.dataset = dataset
        return executor


def main():
    import copy
    dataset = ds.Dataset.all_sports("./inputs/allSportsDataForMatlab.mat")
    parameters = ds.Parameters(
        batch_capacity=30,
        max_workers=10,
        enable_random_rep=False,
        cap=80,
        eta=0.0001,
        delta=0.3,
        phi_function=components.funcs.default_phi_function)
    data = scipy.io.loadmat("./inputs/allSportsDataForMatlab.mat")
    existed_answers = {}
    for i, pair in enumerate(data["PairsY"]):
        left = pair[0]
        right = pair[1]
        ans = list(data["Yans"][i][0][0])
        existed_answers[(left, right)] = ans
        existed_answers[(right, left)] = ans
    executor = BatchQueryExecutor(copy.deepcopy(dataset),
                                  parameters,
                                  existed_answers,
                                  data["Pconf"])
    tracker = executor.execute()
    return tracker


if __name__ == "__main__":
    trackers = []
    for _ in range(10):
        tracker = main()
        trackers.append(tracker.as_pd())
    keys = ["decision_error_rate", "vi_clustered", "median_T", "min_T",
            "max_T", "mean_T", "sd_T", "n_queries", "num_hards", "num_clusters"]
    simulation_result = pd.concat(trackers, axis=1)
    simulation_result.to_pickle("./outputs/all_sports_simulation_raw.pkl")
    simulation_result.loc[keys].to_csv(
        "./outputs/all_sports_simulation_result.csv")
