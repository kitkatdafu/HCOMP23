from __future__ import annotations

import copy
import itertools
import components.ds
import executor
import components.funcs as funcs


class ParametersSet:

    def __init__(self, parameters):
        self.cart = itertools.product(parameters["enable_random_set"],
                                      parameters["cap_set"],
                                      parameters["eta_set"],
                                      parameters["delta_set"],
                                      parameters["batch_capacity"],
                                      parameters["max_workers"])

    def __iter__(self):
        return self

    def __next__(self) -> components.ds.Parameters:
        enable, cap, eta, delta, batch_cap, max_workers = self.cart.__next__()
        return components.ds.Parameters(batch_capacity=batch_cap,
                                        max_workers=max_workers,
                                        cap=cap,
                                        phi_function=funcs.default_phi_function,
                                        eta=eta,
                                        delta=delta,
                                        enable_random_rep=enable)


class DatasetSet:

    def __init__(self, parameters):
        self.card = itertools.product(parameters["n"],
                                      parameters["c_set"],
                                      parameters["p_set"],
                                      parameters["always_hard_p_set"],
                                      parameters["always_hard_ratio_set"])

    def __iter__(self):
        return self

    def __next__(self) -> components.ds.Dataset:
        n, c, p, always_hard_p, always_hard_ratio = self.card.__next__()
        return components.ds.Dataset.dummy(n,
                                           c,
                                           p,
                                           always_hard_p,
                                           always_hard_ratio)


class ExecutorSet:

    def __init__(self, num_repeats, dataset_set, parameters_set):
        self.card = itertools.product(range(num_repeats),
                                      dataset_set,
                                      parameters_set)

    def __iter__(self):
        return self

    def __next__(self) -> executor.BatchQueryExecutor:
        _, dataset, parameters = self.card.__next__()
        return executor.BatchQueryExecutor(dataset=copy.deepcopy(dataset),
                                           parameters=parameters)
