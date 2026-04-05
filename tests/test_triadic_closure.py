import networkx as nx
from src.triadic_closure import triadic_closure_scores

def test_triadic_closure():
    G = nx.Graph()
    G.add_edges_from([
        ('A', 'B'),
        ('B', 'C')
    ])

    scores = triadic_closure_scores(G)

    assert ('A', 'C') in scores
    assert scores[('A', 'C')] == 1


if __name__ == "__main__":
    test_triadic_closure()
    print("Test passed ✅")