from __future__ import annotations

import collections.abc
from typing import Dict, Callable


class ConfidenceInterval(collections.abc.Mapping):
    import components.ds
    ci: Dict[int, float]
    phi_function: Callable

    def __init__(self, parameters: components.ds.Parameters):
        self.ci = {}
        self.phi_function = parameters.phi_function
        self.eta = parameters.eta
        self.delta = parameters.delta

    def update(self, t: int):
        self.ci[t] = self.phi_function(t, self.eta, self.delta)

    def __getitem__(self, t: int):
        return self.ci[t]

    def __len__(self):
        return len(self.ci)

    def __iter__(self):
        return iter(self.ci)


class CumulativeEmpiricalAverage(collections.abc.Mapping):
    import components.result
    cea: Dict[int, float]

    def __init__(self):
        self.cea = {0: 0.0}

    def update(self, t: int, result: components.result.QueryResult):
        self.cea[t] = (t - 1) / t * self.cea[t - 1] + 1 / t * float(result)

    def lower_confidence_bound(self, t: int, ci: ConfidenceInterval):
        return self.cea[t] - ci[t]

    def upper_confidence_bound(self, t: int, ci: ConfidenceInterval):
        return self.cea[t] + ci[t]

    def __getitem__(self, t: int):
        return self.cea[t]

    def __len__(self):
        return len(self.cea)

    def __iter__(self):
        return iter(self.cea)
