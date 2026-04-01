"""
graph_test.py
=============
Unit tests for Phase 02 — Graph Construction
---------------------------------------------
Covers:
    - Email layer (directed + undirected collapse)
    - Monthly snapshot slicing
    - Proximity layer (threshold filtering, dept assignment)
    - Graph integrity checks shared by both layers

Run with:
    pytest graph_test.py -v
"""

import io
import pytest
import pandas as pd
import networkx as nx

from graph.email_layer import (
    build_email_graph,
    collapse_to_undirected,
    build_monthly_snapshots,
    email_graph_summary,
)
from graph.proximity_layer import (
    load_tij,
    load_metadata,
    build_proximity_graph,
    proximity_graph_summary,
    filter_by_dept,
)


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def email_df() -> pd.DataFrame:
    """Four emails across two months, three nodes, two depts."""
    return pd.DataFrame(
        {
            "sender":      ["a@e.com", "a@e.com", "b@e.com", "c@e.com"],
            "receiver":    ["b@e.com", "c@e.com", "c@e.com", "a@e.com"],
            "timestamp":   pd.to_datetime(
                               ["2001-01-05", "2001-01-07", "2001-02-03", "2001-02-10"]
                           ).tz_localize("UTC"),
            "sender_dept": ["trading", "trading", "legal",   "finance"],
            "recv_dept":   ["legal",   "finance",  "finance", "trading"],
        }
    )


@pytest.fixture
def tij_df() -> pd.DataFrame:
    raw = """
1351756800 1 2
1351756800 1 3
1351756820 1 2
1351756820 2 3
1351756840 1 2
"""
    return load_tij(io.StringIO(raw.strip()))


@pytest.fixture
def meta_df() -> pd.DataFrame:
    raw = """
1 Dept_A
2 Dept_A
3 Dept_B
"""
    return load_metadata(io.StringIO(raw.strip()))


# ══════════════════════════════════════════════════════════════════════════════
# Email layer — directed graph
# ══════════════════════════════════════════════════════════════════════════════

class TestEmailDirected:

    def test_is_directed(self, email_df):
        G = build_email_graph(email_df)
        assert G.is_directed()

    def test_node_count(self, email_df):
        G = build_email_graph(email_df)
        # a, b, c → 3 nodes
        assert G.number_of_nodes() == 3

    def test_edge_count(self, email_df):
        G = build_email_graph(email_df)
        # a→b, a→c, b→c, c→a → 4 directed edges
        assert G.number_of_edges() == 4

    def test_edge_weights_aggregated(self, email_df):
        """a→b appears once, weight should be 1."""
        G = build_email_graph(email_df)
        assert G["a@e.com"]["b@e.com"]["weight"] == 1

    def test_dept_attribute_on_nodes(self, email_df):
        G = build_email_graph(email_df)
        assert G.nodes["a@e.com"]["dept"] == "trading"
        assert G.nodes["b@e.com"]["dept"] == "legal"

    def test_dept_map_override(self, email_df):
        dept_map = {"a@e.com": "executive"}
        G = build_email_graph(email_df, dept_map=dept_map)
        assert G.nodes["a@e.com"]["dept"] == "executive"

    def test_timestamp_metadata_on_edges(self, email_df):
        G = build_email_graph(email_df)
        edge = G["a@e.com"]["b@e.com"]
        assert "first_ts" in edge
        assert "last_ts" in edge
        assert edge["first_ts"] <= edge["last_ts"]

    def test_missing_column_raises(self):
        bad_df = pd.DataFrame({"sender": ["x"], "receiver": ["y"]})
        with pytest.raises(ValueError, match="missing columns"):
            build_email_graph(bad_df)

    def test_summary_keys(self, email_df):
        G = build_email_graph(email_df)
        s = email_graph_summary(G)
        for key in ("nodes", "edges", "directed", "avg_weight", "depts", "density"):
            assert key in s


# ══════════════════════════════════════════════════════════════════════════════
# Email layer — undirected collapse
# ══════════════════════════════════════════════════════════════════════════════

class TestEmailUndirected:

    def test_is_undirected(self, email_df):
        G_dir = build_email_graph(email_df)
        G_un  = collapse_to_undirected(G_dir)
        assert not G_un.is_directed()

    def test_node_count_preserved(self, email_df):
        G_dir = build_email_graph(email_df)
        G_un  = collapse_to_undirected(G_dir)
        assert G_un.number_of_nodes() == G_dir.number_of_nodes()

    def test_edge_count_reduced(self, email_df):
        """a↔c and c↔a merge; directed 4 edges → undirected ≤ 4."""
        G_dir = build_email_graph(email_df)
        G_un  = collapse_to_undirected(G_dir)
        assert G_un.number_of_edges() <= G_dir.number_of_edges()

    def test_bidirectional_weights_summed(self, email_df):
        """
        a→c weight=1, c→a weight=1 → undirected a↔c weight should be 2.
        """
        G_dir = build_email_graph(email_df)
        G_un  = collapse_to_undirected(G_dir)
        assert G_un["a@e.com"]["c@e.com"]["weight"] == 2

    def test_dept_attributes_preserved(self, email_df):
        G_dir = build_email_graph(email_df)
        G_un  = collapse_to_undirected(G_dir)
        assert G_un.nodes["a@e.com"]["dept"] == "trading"


# ══════════════════════════════════════════════════════════════════════════════
# Monthly snapshots
# ══════════════════════════════════════════════════════════════════════════════

class TestMonthlySnapshots:

    def test_correct_periods_created(self, email_df):
        snaps = build_monthly_snapshots(email_df)
        assert set(snaps.keys()) == {"2001-01", "2001-02"}

    def test_each_snapshot_is_directed(self, email_df):
        snaps = build_monthly_snapshots(email_df)
        for G in snaps.values():
            assert G.is_directed()

    def test_jan_has_two_edges(self, email_df):
        # Jan: a→b, a→c
        snaps = build_monthly_snapshots(email_df)
        assert snaps["2001-01"].number_of_edges() == 2

    def test_feb_has_two_edges(self, email_df):
        # Feb: b→c, c→a
        snaps = build_monthly_snapshots(email_df)
        assert snaps["2001-02"].number_of_edges() == 2

    def test_no_cross_month_edges(self, email_df):
        snaps = build_monthly_snapshots(email_df)
        # Jan graph should NOT contain Feb's c→a edge
        assert not snaps["2001-01"].has_edge("c@e.com", "a@e.com")


# ══════════════════════════════════════════════════════════════════════════════
# Proximity layer
# ══════════════════════════════════════════════════════════════════════════════

class TestProximityGraph:

    def test_is_undirected(self, tij_df, meta_df):
        G = build_proximity_graph(tij_df, meta_df)
        assert not G.is_directed()

    def test_threshold_filters_weak_pairs(self, tij_df, meta_df):
        """
        Fixture: 1↔2 = 3 windows (60s), 1↔3 = 1 window (20s), 2↔3 = 1 window.
        With threshold=60, only 1↔2 survives.
        """
        G = build_proximity_graph(tij_df, meta_df, min_contact_seconds=60)
        assert G.has_edge(1, 2)
        assert not G.has_edge(1, 3)
        assert not G.has_edge(2, 3)

    def test_weight_is_contact_seconds(self, tij_df, meta_df):
        G = build_proximity_graph(tij_df, meta_df, min_contact_seconds=60)
        assert G[1][2]["weight"] == 60  # 3 windows × 20s

    def test_contact_windows_attribute(self, tij_df, meta_df):
        G = build_proximity_graph(tij_df, meta_df, min_contact_seconds=60)
        assert G[1][2]["contact_windows"] == 3

    def test_dept_assigned_from_metadata(self, tij_df, meta_df):
        G = build_proximity_graph(tij_df, meta_df, min_contact_seconds=60)
        assert G.nodes[1]["dept"] == "Dept_A"
        assert G.nodes[2]["dept"] == "Dept_A"

    def test_missing_dept_falls_back_to_unknown(self, tij_df):
        """Without metadata, all depts should be 'unknown'."""
        G = build_proximity_graph(tij_df, metadata_df=None, min_contact_seconds=20)
        for _, attrs in G.nodes(data=True):
            assert attrs["dept"] == "unknown"

    def test_lower_threshold_includes_more_edges(self, tij_df, meta_df):
        G_strict = build_proximity_graph(tij_df, meta_df, min_contact_seconds=60)
        G_loose  = build_proximity_graph(tij_df, meta_df, min_contact_seconds=20)
        assert G_loose.number_of_edges() >= G_strict.number_of_edges()

    def test_summary_keys(self, tij_df, meta_df):
        G = build_proximity_graph(tij_df, meta_df)
        s = proximity_graph_summary(G)
        for key in ("nodes", "edges", "directed", "min_weight_s", "max_weight_s", "density"):
            assert key in s

    def test_filter_by_dept(self, tij_df, meta_df):
        G = build_proximity_graph(tij_df, meta_df, min_contact_seconds=20)
        sub = filter_by_dept(G, "Dept_A")
        # only nodes 1 and 2 are Dept_A
        assert set(sub.nodes()) == {1, 2}

    def test_no_self_loops(self, tij_df, meta_df):
        G = build_proximity_graph(tij_df, meta_df, min_contact_seconds=20)
        assert nx.number_of_selfloops(G) == 0


# ══════════════════════════════════════════════════════════════════════════════
# Cross-layer integrity
# ══════════════════════════════════════════════════════════════════════════════

class TestCrossLayerIntegrity:

    def test_email_undirected_has_no_self_loops(self, email_df):
        G = collapse_to_undirected(build_email_graph(email_df))
        assert nx.number_of_selfloops(G) == 0

    def test_both_layers_have_weight_attribute(self, email_df, tij_df, meta_df):
        G_email = collapse_to_undirected(build_email_graph(email_df))
        G_prox  = build_proximity_graph(tij_df, meta_df, min_contact_seconds=20)
        for G in (G_email, G_prox):
            for _, _, d in G.edges(data=True):
                assert "weight" in d, f"Missing weight in {G.name}"

    def test_both_layers_have_dept_node_attribute(self, email_df, tij_df, meta_df):
        G_email = build_email_graph(email_df)
        G_prox  = build_proximity_graph(tij_df, meta_df, min_contact_seconds=20)
        for G in (G_email, G_prox):
            for _, attrs in G.nodes(data=True):
                assert "dept" in attrs, f"Missing dept in {G.name}"
