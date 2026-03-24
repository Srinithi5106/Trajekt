# tests/test_temporal_betweenness.py
from src.analysis.temporal_betweenness import temporal_betweenness

def test_basic():
    # A -> B at t=1, B -> C at t=2 — B should have TB > 0
    edges = [('A','B',1), ('B','C',2), ('A','C',3)]
    tb = temporal_betweenness(edges, ['A','B','C'])
    assert tb.get('B', 0) > 0, "B should be a temporal broker"
    print("passed:", tb)

test_basic()