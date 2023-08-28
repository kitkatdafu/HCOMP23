import math


def default_phi_function(t: int, eta: float, delta: float) -> float:
    """default phi function according to
    Korlakai Vinayak, Ramya (2018) Graph Clustering: Algorithms, Analysis and
    Query Design. Dissertation (Ph.D.)
    p.g. 90, equation #2

    Args:
        t (int): number of queries
        eta (float): eta
        delta (float): delta

    Returns:
        float: phi(t)
    """
    return (1 + math.sqrt(eta)) * math.sqrt(((1 + eta) / (2 * t)) * (
            math.log(math.log(t + eta * t) + 1) - math.log(delta)))
