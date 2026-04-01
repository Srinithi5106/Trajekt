from collections import defaultdict, deque

def temporal_betweenness(edges_with_time, nodes):
    """
    edges_with_time : list of (u, v, t) — must be sorted by t
    nodes           : list of nodes to use as sources
    returns         : dict {node: tb_score}
    """
    tb = defaultdict(float)

    for source in nodes:
        earliest = {source: -float('inf')}
        pred     = defaultdict(list)
        sigma    = defaultdict(int)
        sigma[source] = 1
        queue = deque([(source, -float('inf'))])
        visited = []

        # forward pass — time-respecting BFS
        while queue:
            u, t_u = queue.popleft()
            visited.append(u)
            for (a, b, t) in edges_with_time:
                if a != u or t <= t_u:
                    continue
                if b not in earliest or t < earliest[b]:
                    earliest[b] = t
                    sigma[b] = sigma[u]
                    pred[b] = [(u, t_u)]
                    queue.append((b, t))
                elif t == earliest[b]:
                    sigma[b] += sigma[u]
                    pred[b].append((u, t_u))

        # backward pass — accumulate dependencies
        delta = defaultdict(float)
        for w in reversed(visited):
            for (v, _) in pred[w]:
                c = (sigma[v] / (sigma[w] + 1e-9)) * (1 + delta[w])
                delta[v] += c
            if w != source:
                tb[w] += delta[w]

    return dict(tb)


def compute_tb_series(df_email, snapshots):
    """
    df_email  : DataFrame with columns [sender, recipient, timestamp, month]
    snapshots : dict {period_str: nx.DiGraph}
    returns   : DataFrame (nodes x months)
    """
    import pandas as pd

    tb_series = {}
    for period, G_snap in snapshots.items():
        snap_df = df_email[df_email['month'] == period]
        edges_t = sorted(
            zip(snap_df['sender'],
                snap_df['recipient'],
                snap_df['timestamp'].astype('int64')),
            key=lambda x: x[2]
        )
        nodes = list(G_snap.nodes())[:50]  # cap for speed
        tb_series[period] = temporal_betweenness(edges_t, nodes)

    return pd.DataFrame(tb_series).fillna(0)