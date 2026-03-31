import networkx as nx
from itertools import combinations

def triadic_closure_scores(G):
    """
    Returns possible edges with number of common neighbors
    """
    scores = {}

    nodes = list(G.nodes())

    for u, v in combinations(nodes, 2):
        if not G.has_edge(u, v):
            common_neighbors = len(list(nx.common_neighbors(G, u, v)))
            if common_neighbors > 0:
                scores[(u, v)] = common_neighbors

    return scores


def apply_triadic_closure(G, threshold=1):
    """
    Adds edges where common neighbors >= threshold
    """
    scores = triadic_closure_scores(G)

    for (u, v), score in scores.items():
        if score >= threshold:
            G.add_edge(u, v)

    return G